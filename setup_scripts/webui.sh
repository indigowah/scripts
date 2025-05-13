#!/bin/bash

# Install Homebrew if not installed
if ! command -v brew &>/dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Add Homebrew to PATH if not already in PATH
if ! echo "$PATH" | grep -q "/usr/local/bin"; then
    echo "Adding Homebrew to PATH..."
    if [ -f "$HOME/.bashrc" ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >>~/.bashrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    if [ -f "$HOME/.zshrc" ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >>~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# Install pyenv if not installed
if ! command -v pyenv &>/dev/null; then
    echo "Installing pyenv..."
    brew install pyenv
fi

# Install pyenv dependencies
echo "Installing pyenv dependencies..."
brew install openssl readline sqlite3 xz zlib tcl-tk

# Add pyenv to shell configuration if not already added
if ! grep -q "pyenv init" "$HOME/.bashrc" 2>/dev/null; then
    echo 'eval "$(pyenv init --path)"' >>~/.bashrc
    echo 'eval "$(pyenv init -)"' >>~/.bashrc
fi
if ! grep -q "pyenv init" "$HOME/.zshrc" 2>/dev/null; then
    echo 'eval "$(pyenv init --path)"' >>~/.zshrc
    echo 'eval "$(pyenv init -)"' >>~/.zshrc
fi

# Reload shell configuration
if [ -n "$ZSH_VERSION" ]; then
    source "$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    source "$HOME/.bashrc"
fi

# Install Python 3.12.8 using pyenv
if ! pyenv versions | grep -q "3.12.8"; then
    echo "Installing Python 3.12.8..."
    pyenv install 3.12.8
fi

# Create "webui" folder in the home directory
WEBUI_DIR="$HOME/webui"
mkdir -p "$WEBUI_DIR"

# Set up a virtual environment for Python 3.12.8
echo "Setting up virtual environment..."
pyenv local 3.12.8
pyenv exec python -m venv "$WEBUI_DIR/venv"

# Activate the virtual environment and install openwebui
source "$WEBUI_DIR/venv/bin/activate"
pip install --upgrade pip
pip install open-webui
deactivate

# Create the startup script for openwebui
cat >"$WEBUI_DIR/start_openwebui.sh" <<'EOF'
#!/bin/bash
# Initialize Homebrew
eval "$(/opt/homebrew/bin/brew shellenv)"

# Activate the virtual environment
source "$HOME/webui/venv/bin/activate"

# Run openwebui as a background task
open-webui serve &
EOF
chmod +x "$WEBUI_DIR/start_openwebui.sh"

# Create the startup script for Ollama
cat >"$WEBUI_DIR/start_ollama.sh" <<'EOF'
#!/bin/bash
# Run Ollama as a background task
ollama serve &
EOF
chmod +x "$WEBUI_DIR/start_ollama.sh"

echo "Setup complete. Use the scripts in $WEBUI_DIR to start OpenWebUI and Ollama."