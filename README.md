<div style="display: flex; justify-content: center; gap: 40px; align-items: center;">
    <img src="figures/upf_logo.png" width="120">
    <img src="figures/aig_upf.jpeg" width=42>
    <img src="figures/eutopia_logo.png" width="120">
</div>

---

# LaPulga - Jorge and Oriol's Capture-the-Flag Pacman agent

#### An AI-based agent for the Pacman Capture the Flag competition

This repository contains the development of an intelligent Pacman agent designed for competitive gameplay in the Pacman Capture the Flag environment. The agent leverages machine learning and game-theoretic approaches to make strategic decisions about offense, defense, and team coordination.

## Project structure

```plaintext
.
├── scripts/                   # Setup and execution scripts
│   ├── config.sh              # Environment configuration
│   └── play.sh                # Run contest matches
├── pacman-contest/            # Contest framework and baseline agents
├── my_team.py                 # Our agent implementation
├── TEAM.md                    # Team information file (was required)
├── .gitignore
├── LICENSE
├── pyproject.toml             # Project dependencies
└── README.md
```

## Requirements

- Python 3.10 or higher
- `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- All Python dependencies are specified in `pyproject.toml`

## Setup

This project uses `uv` for Python environment management.

1. Clone the repository and navigate to the project directory (use `--recursive` to include contest submodules):

```bash
git clone --recursive <repository-url>
cd pacman-agent
```

2. Create and activate the virtual environment:

```bash
uv sync
source .venv/bin/activate
```

On Windows:

```bash
uv sync
.venv\Scripts\activate
```

### Running the agent

You can quickly set up and run a contest match using the provided scripts:

```bash
scripts/config.sh
scripts/play.sh
```
