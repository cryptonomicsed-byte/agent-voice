"""Authenticated sender extraction for kind:1059 call signals.

agent-phone's ``parse_signal`` returns the signal but not who sent it, and the
outer ``pubkey`` of a proper NIP-17 gift wrap is a throwaway key. Routing the
answer to that key sends it to nobody: the real caller is the signer of the
inner seal. This module returns the signal together with the key that is
actually authenticated for it:

* three-layer NIP-17 (wrap -> seal -> rumor): the seal's signer, and the rumor
  must name the same author (otherwise a sender could re-seal someone else's
  rumor and impersonate them);
* legacy one-layer wrap (what the shipped browser caller emits): the outer
  signer, which here is the real sender.

Every signed layer has its id recomputed, so a valid signature cannot be
carried over to different content.

This belongs upstream in agent-phone as a ``parse_signal_with_sender`` API;
it lives here until that lands so the endpoint stops trusting the outer key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import minipae
from agent_phone import Identity
from agent_phone.signaling import Signal

KIND_GIFT_WRAP = 1059
KIND_SEAL = 13
KIND_RUMOR = 14


@dataclass(frozen=True)
class VerifiedSignal:
    signal: Signal
    sender: bytes          # authenticated x-only pubkey of the real sender
    layers: int            # 3 = NIP-17 gift wrap, 1 = legacy single-layer wrap
    event_id: str          # outer event id, for replay de-duplication


def _verify_signed_event(ev: dict, what: str) -> bytes:
    """Check id and signature of one signed event; return its pubkey bytes."""
    try:
        pubkey = bytes.fromhex(ev["pubkey"])
        claimed_id = ev["id"]
        sig = bytes.fromhex(ev["sig"])
        actual_id = minipae.event_id(ev)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{what} is malformed: {e}") from None
    if len(pubkey) != 32 or claimed_id != actual_id:
        raise ValueError(f"{what} id does not match its content")
    if not minipae.schnorr_verify(bytes.fromhex(actual_id), pubkey, sig):
        raise ValueError(f"{what} signature does not verify")
    return pubkey


def parse_signal_with_sender(event: dict, recipient: Identity) -> VerifiedSignal:
    """Decrypt a call-signal gift wrap and return it with its verified sender.

    Raises ValueError for anything that cannot be authenticated.
    """
    if not isinstance(event, dict) or event.get("kind") != KIND_GIFT_WRAP:
        raise ValueError("not a gift-wrap event")
    if ["p", recipient.pubkey_hex] not in event.get("tags", []):
        raise ValueError("gift wrap not addressed to this recipient")

    wrap_pubkey = _verify_signed_event(event, "gift wrap")
    try:
        inner_str = minipae.nip44_decrypt(
            event["content"], minipae.conversation_key(recipient.seckey, wrap_pubkey))
        inner = json.loads(inner_str)
    except Exception as e:
        raise ValueError(f"gift wrap does not decrypt to JSON: {e}") from None

    if isinstance(inner, dict) and inner.get("kind") == KIND_SEAL:
        seal_pubkey = _verify_signed_event(inner, "seal")
        try:
            rumor = json.loads(minipae.nip44_decrypt(
                inner["content"], minipae.conversation_key(recipient.seckey, seal_pubkey)))
        except Exception as e:
            raise ValueError(f"seal does not decrypt to JSON: {e}") from None
        if not isinstance(rumor, dict) or rumor.get("kind") != KIND_RUMOR:
            raise ValueError("seal does not contain a kind:14 rumor")
        if rumor.get("pubkey") != inner["pubkey"]:
            raise ValueError("rumor author does not match the seal signer")
        payload, sender, layers = rumor.get("content", ""), seal_pubkey, 3
    else:
        payload, sender, layers = inner_str, wrap_pubkey, 1

    try:
        signal = Signal.from_json(payload)
    except Exception as e:
        raise ValueError(f"payload is not a call signal: {e}") from None
    return VerifiedSignal(signal, sender, layers, event["id"])
