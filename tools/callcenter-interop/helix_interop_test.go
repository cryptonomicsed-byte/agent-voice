// SPDX-License-Identifier: Apache-2.0

// Interop check: ai-native-callcenter's own Realtime client, on its gateway
// profile, against the HELIX Realtime Gateway. It lives in agent-voice and is
// copied into the call center's internal/provider to run (see README.md);
// it is not part of that repository.
package provider

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestHelixGatewayInterop(t *testing.T) {
	endpoint := os.Getenv("HELIX_GATEWAY_URL")
	if endpoint == "" {
		t.Skip("HELIX_GATEWAY_URL is not set")
	}
	t.Setenv("REALTIME_API_KEY", "interop-key")
	profile := GatewayProfile()
	profile.Endpoint = endpoint

	client, err := New(profile, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	defer client.Close(ctx)

	err = client.Start(ctx, SessionConfig{
		Instructions: "Greet the caller, then help them.",
		Voice:        "helix:dna_pizza01",
		Tools: []ToolSpec{{
			Name:        "transfer_to_agent",
			Description: "Hand the caller to a person",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"queue":{"type":"string"}}}`),
		}},
		InputFormat:  profile.LinearInput,
		OutputFormat: profile.LinearOutput,
	})
	if err != nil {
		t.Fatalf("start: %v", err)
	}

	// wait reads events until one of the wanted type arrives, failing on an error event.
	audioBytes := 0
	wait := func(want EventType, check func(Event) bool) Event {
		t.Helper()
		for {
			select {
			case ev, ok := <-client.Events():
				if !ok {
					t.Fatalf("events closed while waiting for %s", want)
				}
				if ev.Type == EventTypeAudioDelta {
					audioBytes += len(ev.Audio)
				}
				if ev.Type == EventTypeError {
					t.Fatalf("provider error while waiting for %s: %v", want, ev.Err)
				}
				if ev.Type == want && (check == nil || check(ev)) {
					return ev
				}
			case <-ctx.Done():
				t.Fatalf("timed out waiting for %s", want)
			}
		}
	}

	// 1. The session configures, and the opening turn is spoken.
	wait(EventTypeSessionReady, nil)
	wait(EventTypeResponseStarted, nil)
	opening := wait(EventTypeOutputTranscript, func(e Event) bool { return e.IsFinal })
	wait(EventTypeResponseDone, nil)
	if !strings.Contains(opening.Text, "Santino") {
		t.Fatalf("opening was not grounded on the selected business: %q", opening.Text)
	}
	if audioBytes < 24000 { // at least half a second of PCM16 at 24 kHz
		t.Fatalf("opening carried %d bytes of audio", audioBytes)
	}

	// 2. The caller speaks: one second of tone, then a second of silence, in 20 ms frames.
	frame := func(isLoud bool) []byte {
		out := make([]byte, 960)
		if isLoud {
			for i := 0; i < 480; i++ {
				v := int16(12000 * math.Sin(2*math.Pi*440*float64(i)/24000))
				binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
			}
		}
		return out
	}
	for i := 0; i < 100; i++ {
		if err := client.SendAudio(frame(i < 50)); err != nil {
			t.Fatalf("send audio: %v", err)
		}
		time.Sleep(2 * time.Millisecond)
	}
	wait(EventTypeSpeechStarted, nil)
	heard := wait(EventTypeInputTranscript, func(e Event) bool { return e.IsFinal })
	if heard.Text != "I want to talk to a person" {
		t.Fatalf("transcript: %q", heard.Text)
	}

	// 3. The model asks for the flow's tool; the call center runs it and answers.
	call := wait(EventTypeToolCall, nil)
	if call.ToolName != "transfer_to_agent" || !strings.Contains(call.ToolArgs, "support") {
		t.Fatalf("tool call: %s %s", call.ToolName, call.ToolArgs)
	}
	if err := client.SendToolResult(call.ToolCallID, `{"ok":true}`); err != nil {
		t.Fatalf("tool result: %v", err)
	}
	said := wait(EventTypeOutputTranscript, func(e Event) bool { return e.IsFinal })
	if said.Text != "Connecting you now." {
		t.Fatalf("after the tool result the agent said %q", said.Text)
	}
	wait(EventTypeResponseDone, nil)
}
