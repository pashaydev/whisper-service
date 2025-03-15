#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include "whisper.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <chrono>

using json = nlohmann::json;
namespace fs = std::filesystem;

bool download_model(const std::string& model_name) {
    std::string model_path = "models/" + model_name;
    std::string url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + model_name;

    std::cout << "Downloading model " << model_name << "..." << std::endl;

    std::string cmd;
    #ifdef _WIN32
        cmd = "curl -L " + url + " -o " + model_path;
    #else
        cmd = "curl -L " + url + " -o " + model_path + " || wget " + url + " -O " + model_path;
    #endif

    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Failed to download model. Please download manually." << std::endl;
        return false;
    }

    std::cout << "Model downloaded successfully." << std::endl;
    return true;
}


// Function to read audio file
std::vector<float> read_wav_file(const std::string& audio_path) {
    // Open file in binary mode
    std::ifstream file(audio_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open audio file: " + audio_path);
    }

    // WAV header structure
    struct WavHeader {
        // RIFF header
        char riff_id[4];        // "RIFF"
        uint32_t file_size;     // File size - 8
        char wave_id[4];        // "WAVE"

        // FMT chunk
        char fmt_id[4];         // "fmt "
        uint32_t fmt_size;      // Format data length
        uint16_t format;        // Format type (1 = PCM)
        uint16_t channels;      // Number of channels
        uint32_t sample_rate;   // Sample rate
        uint32_t byte_rate;     // Byte rate
        uint16_t block_align;   // Block alignment
        uint16_t bits_per_sample; // Bits per sample
    };

    // Read the WAV header
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));

    // Verify it's a valid WAV file
    if (strncmp(header.riff_id, "RIFF", 4) != 0 ||
        strncmp(header.wave_id, "WAVE", 4) != 0 ||
        strncmp(header.fmt_id, "fmt ", 4) != 0) {
        throw std::runtime_error("Invalid WAV file format");
    }

    // Check for the data chunk
    char chunk_id[4];
    uint32_t chunk_size;

    // Skip any extra format bytes
    if (header.fmt_size > 16) {
        file.seekg(header.fmt_size - 16, std::ios::cur);
    }

    // Find the data chunk
    while (true) {
        if (!file.read(chunk_id, 4)) {
            throw std::runtime_error("Could not find data chunk in WAV file");
        }

        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (strncmp(chunk_id, "data", 4) == 0) {
            // Found the data chunk
            break;
        }

        // Skip this chunk
        file.seekg(chunk_size, std::ios::cur);
    }

    // Calculate number of samples
    size_t num_samples = chunk_size / (header.bits_per_sample / 8) / header.channels;

    // Validate parameters
    if (header.bits_per_sample != 16 && header.bits_per_sample != 8 && header.bits_per_sample != 32) {
        throw std::runtime_error("Unsupported bits per sample: " + std::to_string(header.bits_per_sample));
    }

    std::cout << "WAV file details: " << std::endl;
    std::cout << "  Channels: " << header.channels << std::endl;
    std::cout << "  Sample rate: " << header.sample_rate << std::endl;
    std::cout << "  Bits per sample: " << header.bits_per_sample << std::endl;
    std::cout << "  Number of samples: " << num_samples << std::endl;

    // Read the audio data
    std::vector<float> samples(num_samples);

    if (header.bits_per_sample == 16) {
        // 16-bit PCM
        std::vector<int16_t> buffer(num_samples * header.channels);
        file.read(reinterpret_cast<char*>(buffer.data()), num_samples * header.channels * sizeof(int16_t));

        // Convert to float and handle multiple channels (convert to mono by averaging)
        for (size_t i = 0; i < num_samples; i++) {
            float sum = 0.0f;
            for (uint16_t c = 0; c < header.channels; c++) {
                sum += buffer[i * header.channels + c] / 32768.0f;  // Normalize to -1.0 to 1.0
            }
            samples[i] = sum / header.channels;  // Average all channels
        }
    }
    else if (header.bits_per_sample == 8) {
        // 8-bit PCM (usually unsigned)
        std::vector<uint8_t> buffer(num_samples * header.channels);
        file.read(reinterpret_cast<char*>(buffer.data()), num_samples * header.channels);

        // Convert to float and handle multiple channels
        for (size_t i = 0; i < num_samples; i++) {
            float sum = 0.0f;
            for (uint16_t c = 0; c < header.channels; c++) {
                // 8-bit PCM is usually unsigned (0-255), convert to -1.0 to 1.0
                sum += (buffer[i * header.channels + c] - 128) / 128.0f;
            }
            samples[i] = sum / header.channels;
        }
    }
    else if (header.bits_per_sample == 32) {
        // 32-bit float
        std::vector<float> buffer(num_samples * header.channels);
        file.read(reinterpret_cast<char*>(buffer.data()), num_samples * header.channels * sizeof(float));

        // Handle multiple channels
        for (size_t i = 0; i < num_samples; i++) {
            float sum = 0.0f;
            for (uint16_t c = 0; c < header.channels; c++) {
                sum += buffer[i * header.channels + c];
            }
            samples[i] = sum / header.channels;
        }
    }

    // Resample if the sample rate is not 16000 Hz
    // (Note: Proper resampling would require a more sophisticated approach)
    if (header.sample_rate != 16000) {
        std::cout << "Warning: WAV file sample rate is not 16kHz. Audio might not be processed correctly." << std::endl;
        std::cout << "Recommend using ffmpeg to convert to 16kHz before processing." << std::endl;
    }

    return samples;
}

// Function to transcribe audio using Whisper
json transcribe_audio(const std::string& audio_path) {
    // Initialize whisper parameters
    whisper_context_params params = whisper_context_default_params();

    // Initialize whisper context with params
    struct whisper_context* ctx = whisper_init_from_file_with_params("models/ggml-base.en.bin", params);
    if (ctx == nullptr) {
        throw std::runtime_error("Failed to initialize whisper context");
    }

    // Set full parameters
    whisper_full_params full_params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    full_params.print_realtime = false;
    full_params.print_progress = true;
    full_params.translate = false;
    full_params.language = "en";
    full_params.n_threads = 4;
    full_params.offset_ms = 0;

    // Read audio samples
    std::vector<float> samples;
    try {
        samples = read_wav_file(audio_path);
    } catch (const std::exception& e) {
        whisper_free(ctx);
        throw std::runtime_error(std::string("Failed to read audio: ") + e.what());
    }

    // Process the audio file
    if (whisper_full(ctx, full_params, samples.data(), samples.size()) != 0) {
        whisper_free(ctx);
        throw std::runtime_error("Failed to process audio");
    }

    // Create JSON response
    json result = json::array();

    // Get the number of segments
    const int n_segments = whisper_full_n_segments(ctx);

    for (int i = 0; i < n_segments; ++i) {
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

        // Convert timestamps to seconds
        double time_start = t0 / 100.0;
        double time_end = t1 / 100.0;

        const char* text = whisper_full_get_segment_text(ctx, i);

        json segment = {
            {"timeStart", time_start},
            {"timeEnd", time_end},
            {"text", std::string(text)}
        };

        result.push_back(segment);
    }

    whisper_free(ctx);
    return result;
}

// Function to execute a command and get its output
std::string exec_command(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Convert audio to the format Whisper expects using ffmpeg
std::string convert_audio(const std::string& input_path, const std::string& output_path) {
    // Use ffmpeg to convert to 16kHz mono WAV
    std::string cmd = "ffmpeg -y -i \"" + input_path + "\" -ar 16000 -ac 1 -c:a pcm_s16le \"" + output_path + "\" 2>&1";

    try {
        std::string output = exec_command(cmd);
        return output_path;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to convert audio: ") + e.what());
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--transcribe" && argc > 2) {
        std::string audio_path = argv[2];
        std::cout << "Transcribing file: " << audio_path << std::endl;

        try {
            std::string wav_path = audio_path + ".wav";
            convert_audio(audio_path, wav_path);
            json result = transcribe_audio(wav_path);

            // Print result to console
            std::cout << result.dump(2) << std::endl;

            // Clean up temp file
            std::remove(wav_path.c_str());

            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    // Create HTTP server
    httplib::Server server;



    // Check if public directory exists
    // std::string public_dir = "./public";
    // if (!fs::exists(public_dir)) {
    //     std::cerr << "Warning: Public directory not found at: " << fs::absolute(public_dir).string() << std::endl;

    //     // Try looking in the build directory
    //     public_dir = "../public";
    //     if (fs::exists(public_dir)) {
    //         std::cout << "Found public directory at: " << fs::absolute(public_dir).string() << std::endl;
    //     } else {
    //         std::cerr << "Error: Could not find public directory. Web interface will not work." << std::endl;
    //         std::cerr << "Please ensure the 'public' directory exists in the correct location." << std::endl;
    //     }
    // } else {
    //     std::cout << "Using public directory at: " << fs::absolute(public_dir).string() << std::endl;
    // }


    // // Set up static file server
    // server.set_mount_point("/", public_dir.c_str());


    // Serve index.html explicitly
    // server.Get("/", [&public_dir](const httplib::Request&, httplib::Response& res) {
    //     std::string index_path = public_dir + "/index.html";
    //     if (fs::exists(index_path)) {
    //         std::ifstream file(index_path);
    //         std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    //         res.set_content(content, "text/html");
    //     } else {
    //         res.status = 404;
    //         res.set_content("Index file not found", "text/plain");
    //     }
    // });


    // Serve a simple embedded landing page
       server.Get("/", [](const httplib::Request&, httplib::Response& res) {
           const char* html = R"(
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>Whisper Transcription Service</title>
       <style>
           * {
               box-sizing: border-box;
               margin: 0;
               padding: 0;
           }
           body {
               font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
               line-height: 1.6;
               color: #333;
               background-color: #f8f9fa;
               padding: 20px;
           }
           .container {
               max-width: 900px;
               margin: 0 auto;
               background-color: white;
               border-radius: 8px;
               box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
               padding: 30px;
           }
           h1 {
               text-align: center;
               margin-bottom: 30px;
               color: #2c3e50;
           }
           h2 {
               margin-bottom: 20px;
               color: #3498db;
               border-bottom: 1px solid #eee;
               padding-bottom: 10px;
           }
           .upload-section {
               margin-bottom: 30px;
           }
           .file-input {
               position: relative;
               margin-bottom: 20px;
               width: 100%;
           }
           .file-input input[type="file"] {
               position: absolute;
               left: 0;
               top: 0;
               opacity: 0;
               width: 100%;
               height: 100%;
               cursor: pointer;
               z-index: 10;
           }
           .file-input label {
               display: inline-block;
               padding: 12px 20px;
               background-color: #f1f1f1;
               color: #333;
               border-radius: 4px;
               cursor: pointer;
               width: 100%;
               text-align: center;
               transition: background-color 0.3s;
           }
           .file-input input[type="file"]:hover + label,
           .file-input label:hover {
               background-color: #e1e1e1;
           }
           .controls {
               display: flex;
               align-items: center;
               gap: 20px;
           }
           button {
               padding: 12px 24px;
               background-color: #3498db;
               color: white;
               border: none;
               border-radius: 4px;
               cursor: pointer;
               font-size: 16px;
               transition: background-color 0.3s;
           }
           button:hover {
               background-color: #2980b9;
           }
           #loading {
               display: flex;
               align-items: center;
               gap: 10px;
           }
           .spinner {
               width: 24px;
               height: 24px;
               border: 3px solid rgba(0, 0, 0, 0.1);
               border-radius: 50%;
               border-top-color: #3498db;
               animation: spin 1s ease-in-out infinite;
           }
           @keyframes spin {
               to { transform: rotate(360deg); }
           }
           .hidden {
               display: none !important;
           }
           #results-section {
               margin-top: 30px;
           }
           .results-controls {
               display: flex;
               gap: 10px;
               margin-bottom: 20px;
           }
           #transcript-view {
               background-color: #f9f9f9;
               padding: 20px;
               border-radius: 4px;
               margin-bottom: 20px;
           }
           .transcript-item {
               margin-bottom: 15px;
               padding-bottom: 15px;
               border-bottom: 1px solid #eee;
           }
           .transcript-item:last-child {
               border-bottom: none;
               margin-bottom: 0;
               padding-bottom: 0;
           }
           .transcript-time {
               font-size: 14px;
               color: #777;
               margin-bottom: 5px;
           }
           .transcript-text {
               font-size: 16px;
           }
           #json-view {
               background-color: #272822;
               color: #f8f8f2;
               padding: 20px;
               border-radius: 4px;
               overflow-x: auto;
           }
           #json-content {
               font-family: 'Courier New', Courier, monospace;
               white-space: pre-wrap;
           }
       </style>
   </head>
   <body>
       <div class="container">
           <h1>Whisper Transcription Service</h1>

           <div class="upload-section">
               <h2>Upload Audio File</h2>
               <form id="upload-form">
                   <div class="file-input">
                       <input type="file" id="audio-file" accept="audio/*" required>
                       <label for="audio-file">Choose an audio file</label>
                   </div>
                   <div class="controls">
                       <button type="submit" id="upload-btn">Transcribe</button>
                       <div id="loading" class="hidden">
                           <div class="spinner"></div>
                           <span>Processing audio...</span>
                       </div>
                   </div>
               </form>
           </div>

           <div id="results-section" class="hidden">
               <h2>Transcription Results</h2>
               <div class="results-controls">
                   <button id="toggle-json">Toggle JSON View</button>
                   <button id="copy-json">Copy JSON</button>
                   <button id="download-json">Download JSON</button>
               </div>

               <div id="transcript-view">
                   <div id="transcript-content"></div>
               </div>

               <div id="json-view" class="hidden">
                   <pre id="json-content"></pre>
               </div>
           </div>
       </div>

       <script>
           document.addEventListener('DOMContentLoaded', function() {
               const uploadForm = document.getElementById('upload-form');
               const audioFileInput = document.getElementById('audio-file');
               const uploadBtn = document.getElementById('upload-btn');
               const loading = document.getElementById('loading');
               const resultsSection = document.getElementById('results-section');
               const transcriptContent = document.getElementById('transcript-content');
               const jsonContent = document.getElementById('json-content');
               const toggleJsonBtn = document.getElementById('toggle-json');
               const copyJsonBtn = document.getElementById('copy-json');
               const downloadJsonBtn = document.getElementById('download-json');
               const transcriptView = document.getElementById('transcript-view');
               const jsonView = document.getElementById('json-view');

               let transcriptionData = null;

               // Update file input label with selected filename
               audioFileInput.addEventListener('change', function() {
                   const fileName = this.files[0] ? this.files[0].name : 'Choose an audio file';
                   this.nextElementSibling.textContent = fileName;
               });

               // Handle form submission
               uploadForm.addEventListener('submit', function(e) {
                   e.preventDefault();

                   if (!audioFileInput.files[0]) {
                       alert('Please select an audio file');
                       return;
                   }

                   const formData = new FormData();
                   formData.append('audio', audioFileInput.files[0]);

                   // Show loading spinner
                   uploadBtn.disabled = true;
                   loading.classList.remove('hidden');
                   resultsSection.classList.add('hidden');

                   // Send request to API
                   fetch('/api/transcribe', {
                       method: 'POST',
                       body: formData
                   })
                   .then(response => {
                       if (!response.ok) {
                           throw new Error('Network response was not ok');
                       }
                       return response.json();
                   })
                   .then(data => {
                       // Save transcription data
                       transcriptionData = data;

                       // Display results
                       displayTranscription(data);

                       // Hide loading spinner
                       loading.classList.add('hidden');
                       uploadBtn.disabled = false;
                       resultsSection.classList.remove('hidden');
                   })
                   .catch(error => {
                       console.error('Error:', error);
                       alert('Error transcribing audio: ' + error.message);

                       // Hide loading spinner
                       loading.classList.add('hidden');
                       uploadBtn.disabled = false;
                   });
               });

               // Display transcription in human-readable format
               function displayTranscription(data) {
                   // Format JSON data
                   jsonContent.textContent = JSON.stringify(data, null, 2);

                   // Clear previous transcript
                   transcriptContent.innerHTML = '';

                   // Create HTML for transcript
                   data.segments.forEach(segment => {
                       const item = document.createElement('div');
                       item.className = 'transcript-item';

                       const time = document.createElement('div');
                       time.className = 'transcript-time';
                       time.textContent = `${formatTime(segment.timeStart)} → ${formatTime(segment.timeEnd)}`;

                       const text = document.createElement('div');
                       text.className = 'transcript-text';
                       text.textContent = segment.text.trim();

                       item.appendChild(time);
                       item.appendChild(text);
                       transcriptContent.appendChild(item);
                   });
               }

               // Format time in MM:SS.ms format
               function formatTime(seconds) {
                   const minutes = Math.floor(seconds / 60);
                   const remainingSeconds = (seconds % 60).toFixed(2);
                   return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(5, '0')}`;
               }

               // Toggle between transcript and JSON views
               toggleJsonBtn.addEventListener('click', function() {
                   if (jsonView.classList.contains('hidden')) {
                       jsonView.classList.remove('hidden');
                       transcriptView.classList.add('hidden');
                       this.textContent = 'Show Transcript';
                   } else {
                       jsonView.classList.add('hidden');
                       transcriptView.classList.remove('hidden');
                       this.textContent = 'Toggle JSON View';
                   }
               });

               // Copy JSON to clipboard
               copyJsonBtn.addEventListener('click', function() {
                   if (transcriptionData) {
                       navigator.clipboard.writeText(JSON.stringify(transcriptionData, null, 2))
                           .then(() => alert('JSON copied to clipboard!'))
                           .catch(err => console.error('Failed to copy: ', err));
                   }
               });

               // Download JSON file
               downloadJsonBtn.addEventListener('click', function() {
                   if (transcriptionData) {
                       const dataStr = JSON.stringify(transcriptionData, null, 2);
                       const blob = new Blob([dataStr], { type: 'application/json' });
                       const url = URL.createObjectURL(blob);

                       const a = document.createElement('a');
                       a.href = url;
                       a.download = 'transcription.json';
                       document.body.appendChild(a);
                       a.click();
                       document.body.removeChild(a);
                       URL.revokeObjectURL(url);
                   }
               });
           });
       </script>
   </body>
   </html>
           )";

           res.set_content(html, "text/html");
       });

    // Handle file uploads for transcription
    server.Post("/api/transcribe", [](const httplib::Request& req, httplib::Response& res) {
        // Enable CORS
        res.set_header("Access-Control-Allow-Origin", "*");

        // Start measuring execution time
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!req.has_file("audio")) {
            res.status = 400;
            res.set_content("No audio file provided", "text/plain");
            return;
        }

        // Get uploaded file
        const auto& file = req.get_file_value("audio");
        std::cout << "Received file: " << file.filename << " (" << file.content.size() << " bytes)" << std::endl;

        // Save to temporary file
        std::string temp_path = "/tmp/audio_" + std::to_string(time(nullptr));
        std::ofstream out(temp_path, std::ios::binary);
        out.write(file.content.c_str(), file.content.size());
        out.close();

        // Execution time breakdown
        double convert_time = 0.0;
        double transcribe_time = 0.0;
        double total_time = 0.0;

        try {
            std::cout << "Converting audio file..." << std::endl;

            // Time the conversion step
            auto convert_start = std::chrono::high_resolution_clock::now();

            // Convert audio to the format Whisper expects
            std::string wav_path = temp_path + ".wav";
            convert_audio(temp_path, wav_path);

            auto convert_end = std::chrono::high_resolution_clock::now();
            convert_time = std::chrono::duration<double>(convert_end - convert_start).count();
            std::cout << "Audio conversion completed in " << convert_time << " seconds." << std::endl;

            std::cout << "Transcribing audio file..." << std::endl;

            // Time the transcription step
            auto transcribe_start = std::chrono::high_resolution_clock::now();

            // Transcribe audio
            json result = transcribe_audio(wav_path);

            auto transcribe_end = std::chrono::high_resolution_clock::now();
            transcribe_time = std::chrono::duration<double>(transcribe_end - transcribe_start).count();

            // Calculate total execution time
            auto end_time = std::chrono::high_resolution_clock::now();
            total_time = std::chrono::duration<double>(end_time - start_time).count();

            std::cout << "Transcription complete in " << transcribe_time << " seconds." << std::endl;
            std::cout << "Total request processing time: " << total_time << " seconds." << std::endl;
            std::cout << "Returning " << result.size() << " segments." << std::endl;

            // Add execution time information to the response
            json response = {
                {"segments", result},
                {"executionTime", {
                    {"convert", convert_time},
                    {"transcribe", transcribe_time},
                    {"total", total_time}
                }}
            };

            // Return JSON response
            res.set_content(response.dump(2), "application/json");

            // Clean up temp files
            std::remove(temp_path.c_str());
            std::remove(wav_path.c_str());
        } catch (const std::exception& e) {
            // Calculate time even for errors
            auto end_time = std::chrono::high_resolution_clock::now();
            total_time = std::chrono::duration<double>(end_time - start_time).count();

            std::cerr << "Error during transcription: " << e.what() << std::endl;
            std::cerr << "Failed after " << total_time << " seconds." << std::endl;

            res.status = 500;
            res.set_content(
                json({
                    {"error", e.what()},
                    {"executionTime", total_time}
                }).dump(),
                "application/json"
            );

            std::remove(temp_path.c_str());
        }
    });

    // Health check endpoint
    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Make sure the models directory exists
    if (!fs::exists("models")) {
        std::cerr << "Models directory not found. Creating..." << std::endl;
        fs::create_directory("models");
    }

    // Check if model exists
    if (!fs::exists("models/ggml-base.en.bin")) {
        std::cout << "Model not found. Attempting to download..." << std::endl;
        if (!download_model("ggml-base.en.bin")) {
            std::cerr << "Please download manually using:" << std::endl;
            std::cerr << "curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -o models/ggml-base.en.bin" << std::endl;
        }
    }

    // Check if ffmpeg is installed
    try {
        exec_command("ffmpeg -version");
    } catch (const std::exception&) {
        std::cerr << "Warning: ffmpeg not found. Audio conversion will not work." << std::endl;
        std::cerr << "Please install ffmpeg to enable audio file processing." << std::endl;
    }

    std::cout << "Starting server on http://localhost:8080" << std::endl;
    std::cout << "Visit http://localhost:8080 in your browser to use the web interface" << std::endl;

    // Start the server with built-in error handling
    if (!server.listen("0.0.0.0", 8080)) {
        std::cerr << "Failed to start server!" << std::endl;
    }


    return 0;
}
