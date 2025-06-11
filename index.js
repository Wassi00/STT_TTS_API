const express = require("express");
const app = express();
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
// const textToSpeech = require("@google-cloud/text-to-speech");
const { GoogleGenAI } = require("@google/genai");
const speech = require("@google-cloud/speech");

const PORT = 3001 || process.env.PORT;

require("dotenv").config();

app.use(
  express.urlencoded({
    extended: true,
  })
);

app.use(express.json());

app.use(cors());

// Multer setup
const upload = multer({ dest: "uploads/" });

if (process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON) {
  const credsPath = path.join(__dirname, "google-credentials.json");
  fs.writeFileSync(credsPath, process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON);
  process.env.GOOGLE_APPLICATION_CREDENTIALS = credsPath;
}

// STT client
const clientSTT = new speech.SpeechClient();

app.post("/stt", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file uploaded." });
  }

  try {
    const audioBytes = fs.readFileSync(req.file.path).toString("base64");

    const audio = {
      content: audioBytes,
    };

    const config = {
      // encoding: "LINEAR16",
      languageCode: "ar-MA", // Moroccan Darija
    };

    const request = {
      audio,
      config,
    };

    const [response] = await clientSTT.recognize(request);

    const transcription = response.results
      .map((result) => result.alternatives[0].transcript)
      .join("\n");

    console.log("Transcription:", transcription);

    const responseAPI = await axios.post(
      process.env.API_BASE_URL + process.env.API_URL_GENERAL_CHAT,
      {
        question: transcription,
      }
    );

    console.log("API Response:", responseAPI.data);
    if (!responseAPI.data || !responseAPI.data.response) {
      return res.status(500).json({ error: "No response from API" });
    }

    res.json({
      transcription: transcription,
      apiResponse: responseAPI.data.response,
    });
  } catch (err) {
    console.error("Google STT Error:", err);
    res.status(500).json({ error: "Speech recognition failed" });
  } finally {
    // Clean up temp file
    fs.unlinkSync(req.file.path);
  }
});

app.post("/sttsql", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file uploaded." });
  }

  try {
    const audioBytes = fs.readFileSync(req.file.path).toString("base64");

    const audio = {
      content: audioBytes,
    };

    const config = {
      // encoding: "LINEAR16",
      languageCode: "ar-MA", // Moroccan Darija
    };

    const request = {
      audio,
      config,
    };

    const [response] = await clientSTT.recognize(request);

    const transcription = response.results
      .map((result) => result.alternatives[0].transcript)
      .join("\n");

    console.log("Transcription:", transcription);

    const responseAPI = await axios.post(
      process.env.API_BASE_URL + process.env.API_URL_DATABASE,
      {
        question: transcription,
      }
    );

    console.log("API Response:", responseAPI.data);
    if (!responseAPI.data || !responseAPI.data.response) {
      return res.status(500).json({ error: "No response from API" });
    }

    res.json({
      transcription: transcription,
      apiResponse: responseAPI.data.response,
    });
  } catch (err) {
    console.error("Google STT Error:", err);
    res.status(500).json({ error: "Speech recognition failed" });
  } finally {
    // Clean up temp file
    fs.unlinkSync(req.file.path);
  }
});

const clientG = new GoogleGenAI({ apiKey: process.env.GEN_API_KEY });

app.post("/tts", async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "No text provided" });

  try {
    const result = await clientG.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [
        { parts: [{ text: `Say in warm Moroccan Darija: "${text}"` }] },
      ],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: "Kore" },
          },
        },
      },
    });

    const data = result.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;

    if (!data) {
      console.error("Gemini Error: No audio data returned from Gemini API.");
      return res
        .status(500)
        .json({ error: "No audio data returned from Gemini API" });
    }

    const audioBuffer = Buffer.from(data, "base64");
    const wavBuffer = pcmToWav(audioBuffer, 24000, 1, 16);
    res.set("Content-Type", "audio/wav");
    res.send(wavBuffer);
  } catch (err) {
    console.error("Gemini Error:", err);
    res.status(500).json({ error: "Failed to generate audio" });
  }
});

function pcmToWav(buffer, sampleRate = 24000, numChannels = 1, bitDepth = 16) {
  const header = Buffer.alloc(44);
  const dataLength = buffer.length;
  // ChunkID "RIFF"
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataLength, 4); // ChunkSize
  header.write("WAVE", 8);
  // Subchunk1ID "fmt "
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16); // Subchunk1Size
  header.writeUInt16LE(1, 20); // AudioFormat (PCM)
  header.writeUInt16LE(numChannels, 22); // NumChannels
  header.writeUInt32LE(sampleRate, 24); // SampleRate
  header.writeUInt32LE((sampleRate * numChannels * bitDepth) / 8, 28); // ByteRate
  header.writeUInt16LE((numChannels * bitDepth) / 8, 32); // BlockAlign
  header.writeUInt16LE(bitDepth, 34); // BitsPerSample
  // Subchunk2ID "data"
  header.write("data", 36);
  header.writeUInt32LE(dataLength, 40); // Subchunk2Size
  return Buffer.concat([header, buffer]);
}

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server Up and running at ${PORT}!`);
});
