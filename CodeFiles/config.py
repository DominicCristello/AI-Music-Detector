# ── config.py ─────────────────────────────────────────────────────────────────
# Fill these in from your Stripe dashboard before running.

# Stripe → Developers → API keys → Secret key
STRIPE_SECRET_KEY = "sk_test_REPLACE_WITH_YOUR_KEY"

# Stripe → Products → (each product) → Pricing → copy the price ID
# These map Stripe price IDs to the plan names used in the app
STRIPE_PRICE_IDS = {
    "price_REPLACE_BASIC_ID":    "Basic",
    "price_REPLACE_PREMIUM_ID":  "Premium",
    "price_REPLACE_BUSINESS_ID": "Business",
}

# Local SQLite database file path — sits next to the script
DB_PATH = "users.db"
