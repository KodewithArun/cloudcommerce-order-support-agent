"""CloudCommerce Order Support - Simple and Clean Streamlit UI."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import streamlit as st

st.set_page_config(
    page_title="CloudCommerce Support",
    page_icon="🛍️",
    layout="centered",
)

API_BASE_URL = "http://127.0.0.1:8000"


def check_api() -> bool:
    """Return True if FastAPI backend is available."""
    try:
        return httpx.get(f"{API_BASE_URL}/health", timeout=1.0).status_code == 200
    except Exception:
        return False


def send_chat(session_id: str, customer_id: str, message: str) -> str:
    """Send message to backend or fallback to in-process service."""
    if check_api():
        resp = httpx.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": session_id, "customer_id": customer_id, "message": message},
            timeout=60.0,
        )
        if resp.status_code == 403:
            return "Access denied: session belongs to another customer."
        resp.raise_for_status()
        return resp.json().get("response", "")
    else:
        from app.chat.service import get_chat_service

        return get_chat_service().chat(
            session_id=session_id, customer_id=customer_id, message=message
        )


def get_pending_approvals() -> list[dict[str, Any]]:
    """Fetch pending refund approvals."""
    if check_api():
        try:
            r = httpx.get(f"{API_BASE_URL}/approvals/pending", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return []
    else:
        from app.orders.service import get_order_service

        return [
            {
                "order_id": o.order_id,
                "customer_id": o.customer_id,
                "item": o.item,
                "price": o.price,
            }
            for o in get_order_service().list_pending_approvals()
        ]


def submit_approval(order_id: str, decision: str, session_id: str) -> str:
    """Submit approval or rejection."""
    if check_api():
        r = httpx.post(
            f"{API_BASE_URL}/approve",
            json={"order_id": order_id, "decision": decision, "session_id": session_id},
            timeout=15.0,
        )
        data = r.json()
        return data.get("resumed_response") or data.get("message", "Decision recorded.")
    else:
        from app.chat.service import get_chat_service
        from app.orders.service import get_order_service

        get_order_service().approve_refund(order_id, decision)
        return get_chat_service().resume(session_id=session_id, decision=decision)


# State management
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{uuid.uuid4().hex[:6]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "customer_id" not in st.session_state:
    st.session_state.customer_id = "C1"

# Sidebar
with st.sidebar:
    st.subheader("CloudCommerce Support")

    customer = st.selectbox("Customer", ["C1", "C2"], index=0 if st.session_state.customer_id == "C1" else 1)
    if customer != st.session_state.customer_id:
        st.session_state.customer_id = customer
        st.session_state.session_id = f"sess-{uuid.uuid4().hex[:6]}"
        st.session_state.messages = []
        st.rerun()

    if st.button("New Chat", use_container_width=True):
        st.session_state.session_id = f"sess-{uuid.uuid4().hex[:6]}"
        st.session_state.messages = []
        st.rerun()

    # Manager approvals if any pending
    pending = get_pending_approvals()
    if pending:
        st.markdown("---")
        st.caption("Manager Approval Needed")
        for item in pending:
            st.info(f"Order **{item['order_id']}**: ${item['price']:.2f}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve", key=f"app_{item['order_id']}", use_container_width=True):
                    result = submit_approval(item["order_id"], "approve", st.session_state.session_id)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    st.rerun()
            with col2:
                if st.button("Reject", key=f"rej_{item['order_id']}", use_container_width=True):
                    result = submit_approval(item["order_id"], "reject", st.session_state.session_id)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    st.rerun()

# Main Chat
st.title("Order Support")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if prompt := st.chat_input("Ask about orders, cancellations, refunds, or policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            reply = send_chat(
                session_id=st.session_state.session_id,
                customer_id=st.session_state.customer_id,
                message=prompt,
            )
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # Rerun so if a refund was requested, sidebar shows the pending approval button
    st.rerun()
