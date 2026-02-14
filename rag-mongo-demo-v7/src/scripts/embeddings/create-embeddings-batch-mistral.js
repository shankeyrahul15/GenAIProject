import { MongoClient } from "mongodb";
import dns from "dns";
import dotenv from "dotenv";
import fs from "fs";
import pLimit from "p-limit";
import { generateBatchEmbeddings } from "../utilities/mistralEmbedding.js";

dotenv.config();

// Fix DNS resolution issue on macOS by using Google's DNS servers
dns.setServers(['8.8.8.8', '8.8.4.4']);

// Configure MongoDB client with SSL options
const client = new MongoClient(process.env.MONGODB_URI, {
  ssl: true,
  tlsAllowInvalidCertificates: true,
  tlsAllowInvalidHostnames: true,
  serverSelectionTimeoutMS: 30000,
  connectTimeoutMS: 30000,
  socketTimeoutMS: 30000,
  maxPoolSize: 20
});

// BATCH PROCESSING CONFIGURATION - Optimized for Mistral AI
const BATCH_SIZE = 50; // Process 50 testcases per batch
const CONCURRENT_LIMIT = 3; // Max 3 concurrent batch API calls
const DELAY_BETWEEN_BATCHES = 1000; // 1000ms delay between batches
const MONGODB_BATCH_SIZE = 100; // Insert 100 documents at once

// Create limiters for different operations
const embeddingLimit = pLimit(CONCURRENT_LIMIT);
const dbLimit = pLimit(3);

/**
 * Generate embeddings for a batch of testcases using Mistral AI
 */
async function generateBatchEmbeddingsForTestcases(testcaseBatch, batchNumber, totalBatches, maxRetries = 3) {
  return embeddingLimit(async () => {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        // Prepare input texts for batch processing
        const inputs = testcaseBatch.map(testcase => `
          ID: ${testcase.id}
          Module: ${testcase.module}
          Title: ${testcase.title}
          Description: ${testcase.description}
          Steps: ${testcase.steps}
          Expected Result: ${testcase.expectedResults}
        `.trim());
        
        console.log(`🚀 [Batch ${batchNumber}/${totalBatches}] Processing ${testcaseBatch.length} testcases with Mistral AI...`);
        
        // Use Mistral AI batch embeddings
        const embeddingResult = await generateBatchEmbeddings(inputs);

        if (!embeddingResult || !embeddingResult.embeddings) {
          throw new Error(`Mistral API error: Invalid response`);
        }

        const embeddings = embeddingResult.embeddings;
        const totalTokens = embeddingResult.usage?.total_tokens || 0;
        const model = embeddingResult.model;
        
        // Estimate cost (Mistral pricing: ~$0.10 per 1M tokens)
        const estimatedCost = (totalTokens / 1000000) * 0.10;
        
        // Map embeddings back to testcases
        const results = testcaseBatch.map((testcase, index) => ({
          testcase,
          embedding: embeddings[index],
          cost: estimatedCost / testcaseBatch.length,
          tokens: Math.round(totalTokens / testcaseBatch.length),
          metadata: {
            model: model,
            cost: estimatedCost / testcaseBatch.length,
            tokens: Math.round(totalTokens / testcaseBatch.length),
            apiSource: 'mistral-batch',
            batchNumber: batchNumber,
            createdAt: new Date()
          }
        }));
        
        console.log(`✅ [Batch ${batchNumber}/${totalBatches}] Success! Cost: $${estimatedCost.toFixed(6)} | Tokens: ${totalTokens}`);
        
        return {
          success: true,
          results: results,
          totalCost: estimatedCost,
          totalTokens: totalTokens,
          batchSize: testcaseBatch.length
        };

      } catch (error) {
        lastError = error;
        if (attempt < maxRetries) {
          const waitTime = Math.min(1000 * Math.pow(2, attempt), 10000);
          console.log(`⚠️ [Batch ${batchNumber}/${totalBatches}] Retry ${attempt}/${maxRetries}: ${error.message}`);
          console.log(`   Waiting ${waitTime}ms before retry...`);
          await new Promise(resolve => setTimeout(resolve, waitTime));
        }
      }
    }

    console.error(`❌ [Batch ${batchNumber}/${totalBatches}] Final failure: ${lastError.message}`);
    return {
      success: false,
      error: lastError.message,
      results: testcaseBatch.map(testcase => ({
        testcase,
        error: lastError.message,
        cost: 0,
        tokens: 0
      })),
      totalCost: 0,
      totalTokens: 0,
      batchSize: testcaseBatch.length
    };
  });
}

/**
 * Optimized batch MongoDB insertion
 */
async function insertTestcasesBatch(collection, batch) {
  return dbLimit(async () => {
    if (batch.length === 0) return { inserted: 0, failed: 0 };

    const documents = batch
      .filter(item => !item.error)
      .map(item => ({
        ...item.testcase,
        embedding: item.embedding,
        createdAt: new Date(),
        embeddingMetadata: item.metadata
      }));

    if (documents.length === 0) {
      return { inserted: 0, failed: batch.length };
    }

    try {
      const result = await collection.insertMany(documents, { 
        ordered: false,
        writeConcern: { w: 1 }
      });
      
      const failed = batch.length - documents.length;
      return { inserted: result.insertedCount, failed };
      
    } catch (error) {
      console.error(`❌ Batch insert failed:`, error.message);
      return { inserted: 0, failed: batch.length };
    }
  });
}

/**
 * Progress tracking with ETA calculation
 */
class ProgressTracker {
  constructor(total) {
    this.total = total;
    this.processed = 0;
    this.startTime = Date.now();
    this.lastUpdate = Date.now();
    this.totalCost = 0;
    this.totalTokens = 0;
  }

  update(processed, cost = 0, tokens = 0) {
    this.processed = processed;
    this.totalCost += cost;
    this.totalTokens += tokens;

    const now = Date.now();
    const elapsed = (now - this.startTime) / 1000;
    const rate = this.processed / elapsed;
    const remaining = this.total - this.processed;
    const eta = remaining / rate;

    if (now - this.lastUpdate > 15000 || this.processed === this.total) {
      console.log(`📊 Progress: ${this.processed}/${this.total} (${(this.processed/this.total*100).toFixed(1)}%) | Rate: ${rate.toFixed(1)}/sec | ETA: ${this.formatTime(eta)} | Cost: $${this.totalCost.toFixed(6)}`);
      this.lastUpdate = now;
    }
  }

  formatTime(seconds) {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${Math.floor(seconds/60)}m ${(seconds%60).toFixed(0)}s`;
    return `${Math.floor(seconds/3600)}h ${Math.floor((seconds%3600)/60)}m`;
  }
}

async function main() {
  const overallStart = Date.now();
  
  try {
    await client.connect();
    const db = client.db(process.env.DB_NAME);
    const collection = db.collection(process.env.COLLECTION_NAME);

    // Load testcases
    const testcases = JSON.parse(fs.readFileSync("src/data/testcases.json", "utf-8"));
    const progress = new ProgressTracker(testcases.length);

    console.log(`🚀 MISTRAL AI BATCH PROCESSING: ${testcases.length} test cases`);
    console.log(`⚡ Using Mistral AI Embedding API for Cost-Effective Processing!`);
    console.log(`⚙️  Configuration:`);
    console.log(`   📦 Batch Size: ${BATCH_SIZE} testcases per API call`);
    console.log(`   🔄 Concurrent Batch Calls: ${CONCURRENT_LIMIT}`);
    console.log(`   💾 MongoDB Batch Size: ${MONGODB_BATCH_SIZE}`);
    console.log(`   ⏰ Delay Between Batch Groups: ${DELAY_BETWEEN_BATCHES}ms`);
    console.log(`   🌐 API: Mistral AI (mistral-embed)`);
    console.log(`   📏 Embedding Dimensions: 1024`);
    console.log(`   🗄️  Database: ${process.env.DB_NAME}`);
    console.log(`   📦 Collection: ${process.env.COLLECTION_NAME}`);
    
    // Create batches for concurrent processing
    const batches = [];
    for (let i = 0; i < testcases.length; i += BATCH_SIZE) {
      batches.push({
        testcases: testcases.slice(i, i + BATCH_SIZE),
        batchNumber: Math.floor(i / BATCH_SIZE) + 1
      });
    }
    
    const totalBatches = batches.length;
    
    const batchGroupsCount = Math.ceil(totalBatches / CONCURRENT_LIMIT);
    const estimatedTimePerBatch = 3;
    const estimatedTotal = (batchGroupsCount * estimatedTimePerBatch + (batchGroupsCount - 1) * DELAY_BETWEEN_BATCHES / 1000) / 60;
    console.log(`   📊 Total Batches: ${totalBatches}`);
    console.log(`   🏃 Batch Groups: ${batchGroupsCount}`);
    console.log(`   ⏱️  Estimated Time: ${estimatedTotal.toFixed(1)} minutes\n`);

    let totalCost = 0;
    let totalTokens = 0;
    let totalInserted = 0;
    let totalFailed = 0;
    let processedCount = 0;

    // Process batches with concurrency control
    for (let i = 0; i < batches.length; i += CONCURRENT_LIMIT) {
      const concurrentBatches = batches.slice(i, i + CONCURRENT_LIMIT);
      
      const batchPromises = concurrentBatches.map(batch => 
        generateBatchEmbeddingsForTestcases(batch.testcases, batch.batchNumber, totalBatches)
      );

      const batchResults = await Promise.allSettled(batchPromises);
      
      for (const batchResult of batchResults) {
        if (batchResult.status === 'fulfilled') {
          const result = batchResult.value;
          
          if (result.success) {
            processedCount += result.batchSize;
            totalCost += result.totalCost;
            totalTokens += result.totalTokens;
            progress.update(processedCount, result.totalCost, result.totalTokens);

            const successfulEmbeddings = result.results.filter(item => !item.error);
            
            if (successfulEmbeddings.length > 0) {
              for (let j = 0; j < successfulEmbeddings.length; j += MONGODB_BATCH_SIZE) {
                const subBatch = successfulEmbeddings.slice(j, j + MONGODB_BATCH_SIZE);
                const insertResult = await insertTestcasesBatch(collection, subBatch);
                totalInserted += insertResult.inserted;
                totalFailed += insertResult.failed;
              }
            }
            
            totalFailed += (result.batchSize - successfulEmbeddings.length);
          } else {
            processedCount += result.batchSize;
            totalFailed += result.batchSize;
            progress.update(processedCount, 0, 0);
            console.error(`❌ Batch failed: ${result.error}`);
          }
        } else {
          console.error(`❌ Batch promise rejected:`, batchResult.reason);
        }
      }

      if (i + CONCURRENT_LIMIT < batches.length) {
        console.log(`⏸️  Waiting ${DELAY_BETWEEN_BATCHES}ms before next batch group...`);
        await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES));
      }
    }

    const totalTime = (Date.now() - overallStart) / 1000;
    const rate = testcases.length / totalTime;

    console.log(`\n🎉 MISTRAL AI BATCH PROCESSING COMPLETE!`);
    console.log(`📊 Final Statistics:`);
    console.log(`   ⏱️  Total Time: ${progress.formatTime(totalTime)}`);
    console.log(`   ⚡ Processing Rate: ${rate.toFixed(1)} testcases/second`);
    console.log(`   📝 Total Test Cases: ${testcases.length}`);
    console.log(`   ✅ Successfully Processed: ${totalInserted}`);
    console.log(`   ❌ Failed: ${totalFailed}`);
    console.log(`   📈 Success Rate: ${((totalInserted / testcases.length) * 100).toFixed(1)}%`);
    console.log(`   💰 Total Cost: $${totalCost.toFixed(6)}`);
    console.log(`   🔢 Total Tokens: ${totalTokens.toLocaleString()}`);
    console.log(`   📊 Average Cost per Test Case: $${(totalCost / testcases.length).toFixed(8)}`);
    console.log(`   📊 Average Tokens per Test Case: ${Math.round(totalTokens / testcases.length)}`);
    console.log(`   💡 Cost Savings: ~90% cheaper than OpenAI embeddings`);

  } catch (err) {
    if (err.response) {
      console.error("❌ Mistral API Error:", err.response.status, err.response.data);
    } else {
      console.error("❌ Error:", err.message);
    }
  } finally {
    await client.close();
  }
}

main();
