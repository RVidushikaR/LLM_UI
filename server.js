import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

console.log("Server starting...");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ limit: "10mb", extended: true }));
app.use(express.static(__dirname));

app.post("/api/llm", async (req, res) => {
  const { block, input, model, text, prompt} = req.body; 
  // 'input' = JSON input from previous block (if arrow connection)
  // 'prompt' used for llama fallback
  if (block=="ai_agent"){
    const { block, input, model, prompt, current_block } = req.body;
    console.log("Executing block:", block, "model:", model, "with input:", input, "prompt:", prompt, "current_block:", current_block);
  } else{
    console.log("Executing block:", block, "model:", model, "with input:", input);
 
  }
  try {

    // Word Embedding Block
    if (block === "word_embedding") {
      // input should contain { sentence: "..." }
      const python = spawn("python", ["backend/word_embedding.py", JSON.stringify(input)]);

      let dataBuffer = "";
      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        try {
          const result = JSON.parse(dataBuffer);
          res.json({ type: "embedding", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;
    }

    if (block === "tokenizer") {
      // include the model in the payload
      const payload = { ...input, model }; // merges input with model

      const python = spawn("python", ["backend/tokenizer.py", JSON.stringify(payload)]);

      let dataBuffer = "";
      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        console.log("Raw dense output:", dataBuffer);
        try {
          const result = JSON.parse(dataBuffer);
          res.json({ type: "tokenizer", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;
    }

    if (block === "encoder") {

      const payload = { ...input, model }; 
      if (model=="BERT"){
        const python = spawn("python", ["backend/BERT.py", JSON.stringify(payload)]);

        let dataBuffer = "";
        python.stdout.on("data", data => dataBuffer += data.toString());
        python.stderr.on("data", err => console.error("Python error:", err.toString()));

        python.on("close", () => {
          console.log("Raw BERT output:", dataBuffer);
          try {
            const result = JSON.parse(dataBuffer);
            res.json({ type: "BERT", ...result });
          } catch (err) {
            console.error("JSON parse error:", err);
            res.status(500).json({ text: "Python output parsing failed" });
          }
        });
      } else if (model=="CLIP"){
        const python = spawn("python", ["backend/CLIP.py"]);

        let dataBuffer = "";

        // send payload via stdin
        python.stdin.write(JSON.stringify(payload));
        python.stdin.end();

        python.stdout.on("data", data => dataBuffer += data.toString());
        python.stderr.on("data", err => console.error("Python error:", err.toString()));

        python.on("close", () => {
          console.log("Raw CLIP output:", dataBuffer);
          try {
            const result = JSON.parse(dataBuffer);
            res.json({ type: "CLIP", ...result });
          } catch (err) {
            console.error("JSON parse error:", err);
            res.status(500).json({ text: "Python output parsing failed" });
          }
        });
      } else if (model=="ViT"){
        const python = spawn("python", ["backend/ViT.py"]);

        let dataBuffer = "";

        // send payload via stdin
        python.stdin.write(JSON.stringify(payload));
        python.stdin.end();

        python.stdout.on("data", data => dataBuffer += data.toString());
        python.stderr.on("data", err => console.error("Python error:", err.toString()));

        python.on("close", () => {
          console.log("Raw ViT output:", dataBuffer);
          try {
            const result = JSON.parse(dataBuffer);
            res.json({ type: "ViT", ...result });
          } catch (err) {
            console.error("JSON parse error:", err);
            res.status(500).json({ text: "Python output parsing failed" });
          }
        });
      }
      return;
    } if (block === "softmax") {
      // input should contain { embeddings: [...], mode: "query|key|value" }
      const python = spawn("python", ["backend/softmax.py"]);

      let dataBuffer = "";

      python.stdin.write(JSON.stringify(input));
      python.stdin.end();

      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        console.log("Raw softmax output:", dataBuffer);
        try {
          const result = JSON.parse(dataBuffer);
          res.json({ type: "softmax", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;
    }

    // Dense Layer Block
    if (block === "dense_layer") {
      // input should contain { embeddings: [...], mode: "query|key|value" }
      const python = spawn("python", ["backend/dense_layer.py", JSON.stringify(input)]);

      let dataBuffer = "";
      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        console.log("Raw dense output:", dataBuffer);
        try {
          const result = JSON.parse(dataBuffer);
          res.json({ type: "dense", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;
    }

    // Split Heads Block
    if (block === "split_heads") {
      // input should contain { dense_output: [...], mode: "query|key|value" }
      const python = spawn("python", ["backend/split_head.py", JSON.stringify(input)]);

      let dataBuffer = "";
      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        console.log("Raw split head output:", dataBuffer);
        try {
          const result = JSON.parse(dataBuffer);
          res.json({ type: "heads", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;
    }
    if (block === "attack") {

      if (model === "Back Door"){
        const python = spawn("python", ["backend/backdoor_attack.py", JSON.stringify(input)]);

      let dataBuffer = "";
      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        console.log("Raw backdoor output:", dataBuffer);
        try {
          const result = JSON.parse(dataBuffer);
          res.json({ type: "Back Door", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;

      } else if (model === "Patch"){
        const python = spawn("python", ["backend/patch_attack.py", JSON.stringify(input)]);

      let dataBuffer = "";
      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        console.log("Raw patch output:", dataBuffer);
        try {
          const result = JSON.parse(dataBuffer);
          res.json({ type: "patch", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;

      }
    }

    if (block === "attack_detect") {
      if (model === "Back Door") {
        const python = spawn("python", ["backend/detect_attack.py"]);

        // send input via stdin
        python.stdin.write(JSON.stringify(input));
        python.stdin.end();

        let dataBuffer = "";
        python.stdout.on("data", data => dataBuffer += data.toString());
        python.stderr.on("data", err => console.error("Python error:", err.toString()));

        python.on("close", () => {
          console.log("Raw backdoor output:", dataBuffer);
          try {
            const result = JSON.parse(dataBuffer);
            res.json({ type: "attack_detect", ...result });
          } catch (err) {
            console.error("JSON parse error:", err);
            res.status(500).json({ text: "Python output parsing failed" });
          }
        });

        return;
      }
    }

    // Attention Block
    if (block === "attention") {
      // input should contain { query_heads: [...], key_heads: [...], value_heads: [...] }
      const python = spawn("python", ["backend/attention.py", JSON.stringify(input)]);

      let dataBuffer = "";
      python.stdout.on("data", data => dataBuffer += data.toString());
      python.stderr.on("data", err => console.error("Python error:", err.toString()));

      python.on("close", () => {
        console.log("Raw attention output:", dataBuffer);
        try {
          const result = JSON.parse(dataBuffer);
          // type "attention" for frontend to render
          res.json({ type: "attention", ...result });
        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "Python output parsing failed" });
        }
      });

      return;
    }

    // LLaMA fallback (if model is llama3)
    if (block=="ai_agent") {
      const payload = {
        prompt,
        input,
        current_block: req.body.current_block || null,
        mode: req.body.mode || "normal"
      };

      const python = spawn("python", ["backend/llama3.py", JSON.stringify(payload)]);

      let dataBuffer = "";
      python.stdout.on("data", data => {dataBuffer += data.toString();});

      python.stderr.on("data", err => {console.error("Python error:", err.toString());});

      python.on("close", () => {console.log("Raw LLaMA output:", dataBuffer);
        
        try {
          const result = JSON.parse(dataBuffer);

          res.json({text: result.text || "No response from LLaMA"});

        } catch (err) {
          console.error("JSON parse error:", err);
          res.status(500).json({ text: "LLaMA Python parsing failed" });
        }
      });

      return;
    }

    // Unsupported block
    res.json({ text: "Unsupported block!!" });

  } catch (err) {
    console.error(err);
    res.status(500).json({ text: err.message });
  }
});

app.listen(8080, "0.0.0.0", () => console.log("Server running on port 8080"));
