# ── stripe_utils.py ───────────────────────────────────────────────────────────
# All Stripe API calls live here. SERVER-SIDE ONLY (uses the secret key).
# Requires: pip install stripe

import stripe
from Config import STRIPE_SECRET_KEY, STRIPE_PRICE_IDS

stripe.api_key = STRIPE_SECRET_KEY


def create_stripe_customer(email: str, name: str = "") -> tuple:
    """
    Create a new Stripe customer.
    Returns (success: bool, customer_id: str | None, error: str | None)
    """
    try:
        customer = stripe.Customer.create(
            email=email.strip().lower(),
            name=name.strip() if name else email.split('@')[0],
        )
        return True, customer.id, None
    except stripe.error.StripeError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Unexpected error: {e}"


def get_plan(stripe_customer_id: str):
    """
    Look up the user's active subscription and return their plan name.
    Returns 'Basic', 'Premium', 'Business', or None if no active subscription.
    """
    if not stripe_customer_id:
        return None
    try:
        subscriptions = stripe.Subscription.list(
            customer=stripe_customer_id,
            status='active',
            limit=1,
        )
        if not subscriptions.data:
            return None

        price_id = subscriptions.data[0]['items']['data'][0]['price']['id']
        return STRIPE_PRICE_IDS.get(price_id, 'Unknown Plan')

    except stripe.error.StripeError as e:
        print(f"[Stripe] Error fetching plan: {e}")
        return None
    except Exception as e:
        print(f"[Stripe] Unexpected error: {e}")
        return None


def get_customer_by_email(email: str):
    """Find an existing Stripe customer by email. Returns customer_id or None."""
    try:
        customers = stripe.Customer.list(email=email.strip().lower(), limit=1)
        if customers.data:
            return customers.data[0].id
        return None
    except stripe.error.StripeError as e:
        print(f"[Stripe] Error looking up customer: {e}")
        return None


def create_checkout_session(stripe_customer_id: str, price_id: str,
                            success_url: str, cancel_url: str) -> tuple:
    """
    Create a Stripe Checkout session so the user can subscribe to a plan.
    Returns (success: bool, checkout_url: str | None, error: str | None)
    """
    try:
        session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return True, session.url, None
    except stripe.error.StripeError as e:
        return False, None, str(e)