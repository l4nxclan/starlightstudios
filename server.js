// server.js
// Free semantic duplicate-suggestion detector for BotGhost.
//
// Flow:
//   1. BotGhost fires a webhook to POST /check-suggestion with { "text": "..." }
//   2. This service turns the text into an embedding via Hugging Face's free
//      Inference API (model: sentence-transformers/all-MiniLM-L6-v2)
//   3. It compares that embedding against everything stored in data.json
//      using cosine similarity
//   4. If similarity > SIMILARITY_THRESHOLD, it responds { duplicate: true, ... }
//   5. Otherwise it stores the new suggestion and responds { duplicate: false }
//
// Storage: a flat JSON file (data.json). Good enough for low/medium traffic.
// Render's free tier disk is ephemeral on redeploy, so if you want suggestions
// to survive redeploys long-term, swap the storage functions for Supabase later
// (marked with TODO below) — but this works fine as-is for most servers.

require('dotenv').config();
const express = require('express');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const HF_TOKEN = process.env.HF_API_TOKEN; // free Hugging Face token
const HF_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
const HF_URL = `https://api-inference.huggingface.co/pipeline/feature-extraction/${HF_MODEL}`;

// Simple shared-secret check so randoms on the internet can't spam your endpoint
const SHARED_SECRET = process.env.SHARED_SECRET || '';

// How similar (0-1) two suggestions need to be to count as duplicates.
// 0.85+ = very likely the same idea reworded. Tune this after testing.
const SIMILARITY_THRESHOLD = parseFloat(process.env.SIMILARITY_THRESHOLD || '0.85');

const DATA_FILE = path.join(__dirname, 'data.json');

// ---------- storage helpers ----------

function loadSuggestions() {
  if (!fs.existsSync(DATA_FILE)) return [];
  try {
    return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  } catch (err) {
    console.error('Failed to read data.json, starting fresh:', err);
    return [];
  }
}

function saveSuggestions(suggestions) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(suggestions, null, 2));
}

// TODO (optional upgrade): replace loadSuggestions/saveSuggestions with
// Supabase calls if you want persistence across redeploys / restarts.

// ---------- embedding + similarity helpers ----------

async function getEmbedding(text) {
  const res = await fetch(HF_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ inputs: text, options: { wait_for_model: true } }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Hugging Face API error (${res.status}): ${errText}`);
  }

  const embedding = await res.json();

  // The feature-extraction pipeline can return a nested array (per-token).
  // If so, mean-pool across tokens to get one fixed-size sentence vector.
  if (Array.isArray(embedding[0])) {
    return meanPool(embedding);
  }
  return embedding;
}

function meanPool(tokenVectors) {
  const dim = tokenVectors[0].length;
  const pooled = new Array(dim).fill(0);
  for (const vec of tokenVectors) {
    for (let i = 0; i < dim; i++) pooled[i] += vec[i];
  }
  return pooled.map((v) => v / tokenVectors.length);
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ---------- routes ----------

app.post('/check-suggestion', async (req, res) => {
  try {
    if (SHARED_SECRET && req.headers['x-shared-secret'] !== SHARED_SECRET) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const { text, id } = req.body;
    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'Missing "text" field' });
    }

    const newEmbedding = await getEmbedding(text);
    const suggestions = loadSuggestions();

    let bestMatch = null;
    let bestScore = 0;

    for (const s of suggestions) {
      const score = cosineSimilarity(newEmbedding, s.embedding);
      if (score > bestScore) {
        bestScore = score;
        bestMatch = s;
      }
    }

    const isDuplicate = bestScore >= SIMILARITY_THRESHOLD;

    if (!isDuplicate) {
      suggestions.push({
        id: id || Date.now().toString(),
        text,
        embedding: newEmbedding,
        createdAt: new Date().toISOString(),
      });
      saveSuggestions(suggestions);
    }

    return res.json({
      duplicate: isDuplicate,
      similarity: Number(bestScore.toFixed(3)),
      matched_text: isDuplicate ? bestMatch.text : null,
      matched_id: isDuplicate ? bestMatch.id : null,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

// Health check — useful for Render and for waking a sleeping free instance
app.get('/', (req, res) => {
  res.json({ status: 'ok', storedSuggestions: loadSuggestions().length });
});

// Optional: wipe stored suggestions (protect with the shared secret)
app.post('/reset', (req, res) => {
  if (SHARED_SECRET && req.headers['x-shared-secret'] !== SHARED_SECRET) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  saveSuggestions([]);
  res.json({ status: 'cleared' });
});

app.listen(PORT, () => {
  console.log(`Duplicate detector listening on port ${PORT}`);
});
