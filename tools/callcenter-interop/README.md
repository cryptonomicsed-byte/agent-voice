# Interop check: ai-native-callcenter ↔ HELIX Realtime Gateway

Runs the call center's own Realtime client (`internal/provider`, gateway
profile) against `agent_voice.realtime_gateway`. It proves the two agree on
the wire: session setup, per-call agent selection by the flow's voice, the
opening turn, caller audio and transcription, a tool call and its result, and
the spoken reply.

The Go test has to live inside the call center's tree to reach its internal
package, so it is copied in for the run and removed afterwards. It is not
part of that repository.

```bash
# 1. gateway with scripted speech models (no API keys needed)
python tools/callcenter-interop/serve_fake_models.py 18080 &

# 2. the call center's client against it
CC=/path/to/ai-native-callcenter
cp tools/callcenter-interop/helix_interop_test.go $CC/internal/provider/
(cd $CC && HELIX_GATEWAY_URL=ws://127.0.0.1:18080/v1/realtime \
    go test -count=1 -run TestHelixGatewayInterop ./internal/provider/)
rm $CC/internal/provider/helix_interop_test.go
```

Last run: 2026-11-14, ai-native-callcenter `f052b35`, result `ok`.

What this does not cover: a real SIP call through FreeSWITCH, real speech
models, and the call center's full call actor (only its provider client).
