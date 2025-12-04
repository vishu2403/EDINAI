import React, { useEffect, useMemo, useRef, useState } from 'react';

const ICE_CONFIGURATION = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

const buildWebSocketUrl = (roomId, clientId) => {
  const { protocol, host } = window.location;
  const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${host}/api/school-portal/ws/webrtc/${encodeURIComponent(roomId)}?clientId=${encodeURIComponent(
    clientId
  )}`;
};

const statusMessages = {
  connecting: 'Connecting…',
  awaitingPeer: 'Waiting for another participant…',
  inCall: 'Call in progress',
  disconnected: 'Disconnected',
  error: 'Unable to establish call',
};

const StudentPortalCall = ({ roomId, clientId, onClose }) => {
  const [status, setStatus] = useState('connecting');
  const [error, setError] = useState(null);
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const websocketRef = useRef(null);
  const localStreamRef = useRef(null);
  const makingOfferRef = useRef(false);
  const politeRef = useRef(false);

  const sendSignal = useMemo(() => {
    return (payload) => {
      const websocket = websocketRef.current;
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(payload));
      }
    };
  }, []);

  useEffect(() => {
    let isMounted = true;

    const cleanup = () => {
      makingOfferRef.current = false;

      if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.close();
      }
      websocketRef.current = null;

      if (peerConnectionRef.current) {
        peerConnectionRef.current.onicecandidate = null;
        peerConnectionRef.current.ontrack = null;
        peerConnectionRef.current.onconnectionstatechange = null;
        peerConnectionRef.current.close();
      }
      peerConnectionRef.current = null;

      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => track.stop());
        localStreamRef.current = null;
      }

      if (localVideoRef.current) {
        localVideoRef.current.srcObject = null;
      }
      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = null;
      }
    };

    const initialise = async () => {
      try {
        setStatus('connecting');

        const localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        if (!isMounted) {
          localStream.getTracks().forEach((track) => track.stop());
          return;
        }

        localStreamRef.current = localStream;
        if (localVideoRef.current) {
          localVideoRef.current.srcObject = localStream;
        }

        const peerConnection = new RTCPeerConnection(ICE_CONFIGURATION);
        peerConnectionRef.current = peerConnection;

        localStream.getTracks().forEach((track) => peerConnection.addTrack(track, localStream));

        peerConnection.onicecandidate = (event) => {
          if (event.candidate) {
            sendSignal({ type: 'candidate', candidate: event.candidate });
          }
        };

        peerConnection.ontrack = (event) => {
          const [stream] = event.streams;
          if (remoteVideoRef.current) {
            remoteVideoRef.current.srcObject = stream;
          }
          setStatus('inCall');
        };

        peerConnection.onconnectionstatechange = () => {
          const { connectionState } = peerConnection;
          if (connectionState === 'disconnected' || connectionState === 'failed') {
            setStatus('disconnected');
          }
        };

        const websocket = new WebSocket(buildWebSocketUrl(roomId, clientId));
        websocketRef.current = websocket;

        websocket.onopen = () => {
          setStatus('awaitingPeer');
        };

        websocket.onmessage = async (event) => {
          try {
            const payload = JSON.parse(event.data);
            if (payload.sender === clientId) {
              return;
            }

            switch (payload.type) {
              case 'peer-joined':
                politeRef.current = true;
                if (!makingOfferRef.current) {
                  makingOfferRef.current = true;
                  const offer = await peerConnection.createOffer();
                  await peerConnection.setLocalDescription(offer);
                  sendSignal({ type: 'offer', sdp: offer.sdp });
                  makingOfferRef.current = false;
                  setStatus('connecting');
                }
                break;
              case 'offer':
                politeRef.current = true;
                await peerConnection.setRemoteDescription({ type: 'offer', sdp: payload.sdp });
                const answer = await peerConnection.createAnswer();
                await peerConnection.setLocalDescription(answer);
                sendSignal({ type: 'answer', sdp: answer.sdp });
                setStatus('connecting');
                break;
              case 'answer':
                await peerConnection.setRemoteDescription({ type: 'answer', sdp: payload.sdp });
                setStatus('connecting');
                break;
              case 'candidate':
                if (payload.candidate) {
                  await peerConnection.addIceCandidate(payload.candidate);
                }
                break;
              case 'peer-left':
                if (remoteVideoRef.current) {
                  remoteVideoRef.current.srcObject = null;
                }
                setStatus('awaitingPeer');
                break;
              default:
                break;
            }
          } catch (receivedError) {
            console.error('WebRTC signaling error:', receivedError);
            setError(receivedError);
            setStatus('error');
          }
        };

        websocket.onclose = () => {
          setStatus('disconnected');
        };

        websocket.onerror = () => {
          setStatus('error');
        };
      } catch (initialiseError) {
        console.error('Failed to start call', initialiseError);
        setError(initialiseError);
        setStatus('error');
      }
    };

    initialise();

    return () => {
      isMounted = false;
      cleanup();
    };
  }, [roomId, clientId, sendSignal]);

  const handleEndCall = () => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.close();
    }
    if (onClose) {
      onClose();
    }
  };

  const statusLabel = statusMessages[status] ?? statusMessages.connecting;

  return (
    <div className="rounded-2xl border border-white/10 bg-[#141825] p-4">
      <div className="mb-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-white/40">Video Call</p>
          <p className="text-sm font-medium text-white/80">{statusLabel}</p>
          {error ? (
            <p className="text-xs text-red-400">{error.message || 'Unexpected error occurred.'}</p>
          ) : null}
        </div>
        <button
          type="button"
          onClick={handleEndCall}
          className="w-full rounded-full bg-red-500/90 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white transition hover:bg-red-400 md:w-auto"
        >
          End Call
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="relative overflow-hidden rounded-2xl bg-black/50">
          <video
            ref={localVideoRef}
            autoPlay
            muted
            playsInline
            className="h-48 w-full object-cover sm:h-56"
          />
          <span className="absolute bottom-2 left-2 rounded-full bg-black/60 px-3 py-1 text-xs uppercase tracking-[0.2em] text-white/80">
            You
          </span>
        </div>
        <div className="relative overflow-hidden rounded-2xl bg-black/30">
          <video
            ref={remoteVideoRef}
            autoPlay
            playsInline
            className="h-48 w-full object-cover sm:h-56"
          />
          <span className="absolute bottom-2 left-2 rounded-full bg-black/60 px-3 py-1 text-xs uppercase tracking-[0.2em] text-white/80">
            Participant
          </span>
        </div>
      </div>
    </div>
  );
};

export default StudentPortalCall;
