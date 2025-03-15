#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include "whisper.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// These functions should match those in main.cpp
// You could also extract these to separate files to avoid duplication
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <audio_file> [output_file]" << std::endl;
        std::cerr << "  If output_file is not specified, output is printed to stdout" << std::endl;
        return 1;
    }

    // Check if model exists
    if (!fs::exists("models/ggml-base.en.bin")) {
        std::cerr << "Model not found at models/ggml-base.en.bin" << std::endl;
        std::cerr << "Please download manually using:" << std::endl;
        std::cerr << "curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -o models/ggml-base.en.bin" << std::endl;
        return 1;
    }

    // Check if ffmpeg is installed
    try {
        exec_command("ffmpeg -version");
    } catch (const std::exception&) {
        std::cerr << "Error: ffmpeg not found. Audio conversion will not work." << std::endl;
        std::cerr << "Please install ffmpeg to enable audio file processing." << std::endl;
        return 1;
    }

    std::string audio_path = argv[1];
    std::cout << "Transcribing file: " << audio_path << std::endl;

    try {
        std::string temp_path = "/tmp/audio_" + std::to_string(time(nullptr));
        std::string wav_path = temp_path + ".wav";

        // Convert audio to WAV format
        std::cout << "Converting audio..." << std::endl;
        convert_audio(audio_path, wav_path);

        // Transcribe audio
        std::cout << "Transcribing audio..." << std::endl;
        json result = transcribe_audio(wav_path);

        // Output the result
        if (argc > 2) {
            // Write to specified output file
            std::string output_file = argv[2];
            std::ofstream out(output_file);
            if (!out.is_open()) {
                std::cerr << "Error: Could not open output file: " << output_file << std::endl;
                return 1;
            }
            out << result.dump(2);
            out.close();
            std::cout << "Transcription saved to: " << output_file << std::endl;
        } else {
            // Print to stdout
            std::cout << result.dump(2) << std::endl;
        }

        // Clean up temporary files
        std::remove(wav_path.c_str());

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
