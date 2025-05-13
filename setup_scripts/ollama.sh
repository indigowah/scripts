#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Homebrew if not installed
if ! command_exists brew; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Install Ollama using Homebrew
if ! brew list ollama &>/dev/null; then
    echo "Installing Ollama..."
    brew install ollama
else
    echo "Ollama is already installed."
fi

# Create a scripts folder in the home directory
SCRIPTS_DIR="$HOME/scripts"
if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "Creating scripts directory at $SCRIPTS_DIR..."
    mkdir -p "$SCRIPTS_DIR"
fi

# Create a startup script to run Ollama serve in the background
STARTUP_SCRIPT="$SCRIPTS_DIR/start_ollama.sh"
cat << 'EOF' > "$STARTUP_SCRIPT"
#!/bin/bash
# Start Ollama serve as a background task with no terminal
nohup ollama serve >/dev/null 2>&1 &
EOF

# Make the startup script executable
chmod +x "$STARTUP_SCRIPT"

echo "Setup complete. You can run the startup script using: $STARTUP_SCRIPT"