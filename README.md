# Sales Ai

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 📖 Description

An AI app to find real-time discounts/deals/sales prices from various online markets around the world. Will be modified to use the latest ChatGPT API and dynamically capture user trends. This is a work in progress.

## ✨ Features

- Real-time price monitoring
- AI-powered deal detection
- Multi-market price comparison
- User trend analysis
- Automated discount alerts
- Price history tracking

## 🛠️ Tech Stack

- **Primary Language:** Python
- **Framework:** Ai/Ml
- **Additional Tools:** N/A

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/shivajipanam/sales-ai.git
cd sales-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# Start the application
docker-compose up -d
``````bash
# Start the sales AI system
python main.py

# Monitor specific products
python monitor.py --product "laptop" --price 1000

# Get trending deals
python trends.py --category "electronics"
```