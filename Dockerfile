FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libssl-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone required repositories
RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
    git clone https://github.com/yhirose/cpp-httplib.git && \
    git clone https://github.com/nlohmann/json.git

# Copy source files
COPY main.cpp .
COPY cli.cpp .
COPY CMakeLists.txt .
COPY public ./public

# Create models directory
RUN mkdir -p models

# Download Whisper model
RUN curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin \
    -o models/ggml-base.en.bin

# Build the application
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Expose port for web service
EXPOSE 8080

# Set working directory to build folder
WORKDIR /app/build

# By default run the web service
CMD ["./whisper_service"]
