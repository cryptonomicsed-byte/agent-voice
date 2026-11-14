"""Sender authentication for call signals (audit P0-1, acceptance T04/T05).

A proper NIP-17 gift wrap is signed by a throwaway key, so the endpoint must
route on the seal signer. The legacy one-layer wrap the browser caller emits
is signed by the real sender. Both are tested independently.
"""
import json

import minipae
import pytest
from agent_phone import Identity, SignalType
from agent_phone.signaling import (Signal, build_gift_wrap, build_hangup,
                                   build_offer, build_signal_wrap)

from agent_voice.endpoint import VoiceEndpoint
from agent_voice.verified_signal import parse_signal_with_sender

CALL = "c" * 32


def _offer_json(call_id=CALL):
    return Signal(SignalType.OFFER, call_id, sdp="v=0").to_json()


def test_three_layer_sender_is_seal_signer_not_outer_key():
    caller, callee = Identity.generate(), Identity.generate()
    ev = build_offer(caller, callee.pubkey, "v=0", CALL)
    assert ev["pubkey"] != caller.pubkey_hex          # outer key is a throwaway
    v = parse_signal_with_sender(ev, callee)
    assert v.layers == 3
    assert v.sender == caller.pubkey
    assert v.signal.type is SignalType.OFFER and v.signal.call_id == CALL


def test_one_layer_sender_is_outer_signer():
    caller, callee = Identity.generate(), Identity.generate()
    ev = build_signal_wrap(caller, callee.pubkey, _offer_json())
    v = parse_signal_with_sender(ev, callee)
    assert v.layers == 1
    assert v.sender == caller.pubkey


def test_rejects_wrong_recipient():
    caller, callee, other = (Identity.generate() for _ in range(3))
    ev = build_offer(caller, callee.pubkey, "v=0", CALL)
    with pytest.raises(ValueError, match="not addressed"):
        parse_signal_with_sender(ev, other)


def test_rejects_bad_outer_signature():
    caller, callee = Identity.generate(), Identity.generate()
    ev = build_offer(caller, callee.pubkey, "v=0", CALL)
    ev["sig"] = "00" * 64
    with pytest.raises(ValueError, match="signature"):
        parse_signal_with_sender(ev, callee)


def test_rejects_signature_moved_onto_other_content():
    caller, callee = Identity.generate(), Identity.generate()
    a = build_signal_wrap(caller, callee.pubkey, _offer_json())
    b = build_signal_wrap(caller, callee.pubkey, _offer_json("d" * 32))
    forged = dict(a, content=b["content"])            # a's id+sig, b's content
    with pytest.raises(ValueError, match="id does not match"):
        parse_signal_with_sender(forged, callee)


def _wrap_seal(seal: dict, callee: Identity) -> dict:
    throwaway = Identity.generate()
    return minipae.sign_event(
        1059,
        minipae.nip44_encrypt(json.dumps(seal),
                              minipae.conversation_key(throwaway.seckey, callee.pubkey)),
        [["p", callee.pubkey_hex]], throwaway.seckey)


def _seal(signer: Identity, callee: Identity, rumor_author_hex: str) -> dict:
    rumor = {"kind": 14, "content": _offer_json(), "tags": [["p", callee.pubkey_hex]],
             "created_at": 0, "pubkey": rumor_author_hex}
    return minipae.sign_event(
        13,
        minipae.nip44_encrypt(json.dumps(rumor),
                              minipae.conversation_key(signer.seckey, callee.pubkey)),
        [], signer.seckey)


def test_rejects_forged_seal_signature():
    caller, callee = Identity.generate(), Identity.generate()
    seal = _seal(caller, callee, caller.pubkey_hex)
    seal["sig"] = "00" * 64
    with pytest.raises(ValueError, match="seal signature"):
        parse_signal_with_sender(_wrap_seal(seal, callee), callee)


def test_rejects_rumor_claiming_another_author():
    attacker, victim, callee = (Identity.generate() for _ in range(3))
    seal = _seal(attacker, callee, victim.pubkey_hex)
    with pytest.raises(ValueError, match="rumor author"):
        parse_signal_with_sender(_wrap_seal(seal, callee), callee)


def test_rejects_non_signal_payload():
    caller, callee = Identity.generate(), Identity.generate()
    ev = build_gift_wrap(caller, callee.pubkey, "hello, this is a chat DM")
    with pytest.raises(ValueError, match="not a call signal"):
        parse_signal_with_sender(ev, callee)


# -- endpoint routing --------------------------------------------------------

class _Endpoint(VoiceEndpoint):
    """VoiceEndpoint with media stubbed out; records routing decisions."""

    def __init__(self, identity):
        self.identity = identity
        self.sessions, self.peer_connections = {}, {}
        self._seen_events, self._call_started = {}, {}
        self.card_memory = None
        self.answered, self.closed = [], []

    async def _answer_offer(self, sig, caller_pubkey):
        from agent_phone.signaling import CallSession
        s = CallSession(self.identity)
        s.on_signal(sig, caller_pubkey)
        self.sessions[sig.call_id] = s
        self.answered.append((sig.call_id, caller_pubkey))

    async def _close_call(self, call_id):
        self.sessions.pop(call_id, None)
        self.closed.append(call_id)


async def test_endpoint_answers_three_layer_offer_to_real_caller():
    caller, callee = Identity.generate(), Identity.generate()
    ep = _Endpoint(callee)
    await ep._handle_event(build_offer(caller, callee.pubkey, "v=0", CALL))
    assert ep.answered == [(CALL, caller.pubkey)]


async def test_endpoint_ignores_replayed_offer():
    caller, callee = Identity.generate(), Identity.generate()
    ep = _Endpoint(callee)
    ev = build_offer(caller, callee.pubkey, "v=0", CALL)
    await ep._handle_event(ev)
    await ep._handle_event(ev)                                    # same event, second relay
    await ep._handle_event(build_offer(caller, callee.pubkey, "v=0", CALL))  # re-wrapped
    assert len(ep.answered) == 1


async def test_endpoint_hangup_only_from_the_caller():
    caller, callee, stranger = (Identity.generate() for _ in range(3))
    ep = _Endpoint(callee)
    await ep._handle_event(build_offer(caller, callee.pubkey, "v=0", CALL))
    await ep._handle_event(build_hangup(stranger, callee.pubkey, CALL))
    assert ep.closed == [] and CALL in ep.sessions
    await ep._handle_event(build_hangup(caller, callee.pubkey, CALL))
    assert ep.closed == [CALL]
