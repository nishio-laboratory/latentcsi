import { useEffect, useRef, useState, type ChangeEvent } from "react";
import ImageDisplay from "./components/ImageDisplay";
import IntervalSlider from "./components/IntervalSlider";
import ControlButtons from "./components/ControlButtons";
import { SDSettings } from "./components/SDSettings";
import StatusDisplay from "./components/StatusDisplay";
import type { TrainerState } from "./components/StatusDisplay";
import MessageMenu from "./components/MessageMenu";

const RECONNECT_DELAY_MS = 3000;
const IMAGE_SIZE = 375;

type ImageChannel = "pred" | "true";

type ImageMessage = {
  type: "image";
  channel: ImageChannel;
  img: string;
};

type TrainerStatusMessage = {
  type: "trainer_status";
  status: TrainerState;
};

type ConnectionStateKind =
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";
type ConnectionScope = "frontend" | "server";

type ConnectionStateMessage = {
  type: "connection_state";
  scope: ConnectionScope;
  state: ConnectionStateKind;
  detail?: string | null;
};

type ErrorMessage = {
  type: "error";
  message: string;
  detail?: string | null;
};

type ServerMessage =
  | ImageMessage
  | TrainerStatusMessage
  | ConnectionStateMessage
  | ErrorMessage;

type ConnectionStatus = "idle" | ConnectionStateKind;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseServerMessage(raw: unknown): ServerMessage | null {
  if (!isRecord(raw) || typeof raw.type !== "string") {
    return null;
  }

  switch (raw.type) {
    case "image": {
      const { channel, img } = raw as {
        channel?: unknown;
        img?: unknown;
      };
      if (
        (channel === "pred" || channel === "true") &&
        typeof img === "string"
      ) {
        return { type: "image", channel, img };
      }
      return null;
    }

    case "trainer_status": {
      const { status } = raw as { status?: unknown };
      if (!isRecord(status)) {
        return null;
      }
      const statusRaw = status as Record<string, unknown>;
      const { started, training, recording, batches_trained, reservoir_size } =
        statusRaw;
      if (
        typeof started === "boolean" &&
        typeof training === "boolean" &&
        typeof recording === "boolean" &&
        typeof batches_trained === "number" &&
        typeof reservoir_size === "number"
      ) {
        return {
          type: "trainer_status",
          status: {
            started,
            training,
            recording,
            batches_trained,
            reservoir_size,
          },
        };
      }
      return null;
    }

    case "connection_state": {
      const { scope, state, detail } = raw as {
        scope?: unknown;
        state?: unknown;
        detail?: unknown;
      };
      if (scope !== "frontend" && scope !== "server") {
        return null;
      }
      if (
        state === "connecting" ||
        state === "connected" ||
        state === "disconnected" ||
        state === "error"
      ) {
        const detailValue =
          typeof detail === "string"
            ? detail
            : detail === null
              ? null
              : undefined;
        return detailValue === undefined
          ? { type: "connection_state", scope, state }
          : { type: "connection_state", scope, state, detail: detailValue };
      }
      return null;
    }

    case "error": {
      const { message, detail } = raw as {
        message?: unknown;
        detail?: unknown;
      };
      if (typeof message !== "string") {
        return null;
      }
      const detailValue =
        typeof detail === "string"
          ? detail
          : detail === null
            ? null
            : undefined;
      return detailValue === undefined
        ? { type: "error", message }
        : { type: "error", message, detail: detailValue };
    }

    default:
      return null;
  }
}

function App() {
  const [predImageSrc, setPredImageSrc] = useState<string>("");
  const [trueImageSrc, setTrueImageSrc] = useState<string>("");
  const [intervalValue, setIntervalValue] = useState<number>(0.2);
  const [showTrue, setShowTrue] = useState<boolean>(false);
  const [trainerState, setTrainerState] = useState<TrainerState | null>(null);
  const [frontendStatus, setFrontendStatus] =
    useState<ConnectionStateKind>("connecting");
  const [frontendDetail, setFrontendDetail] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<ConnectionStatus>("idle");
  const [serverDetail, setServerDetail] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let shouldReconnect = true;
    let reconnectTimeout: number | null = null;

    function scheduleReconnect(detail: string | null) {
      if (!shouldReconnect) {
        return;
      }
      setFrontendStatus("disconnected");
      setFrontendDetail(detail);
      setServerStatus("disconnected");
      setServerDetail(detail);
      setPredImageSrc("");
      setTrueImageSrc("");
      setTrainerState(null);
      ws.current = null;
      if (reconnectTimeout !== null) {
        window.clearTimeout(reconnectTimeout);
      }
      reconnectTimeout = window.setTimeout(() => {
        if (!shouldReconnect) {
          return;
        }
        setFrontendStatus("connecting");
        setFrontendDetail(null);
        connect();
      }, RECONNECT_DELAY_MS);
    }

    function connect() {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
      ws.current = socket;

      socket.onopen = () => {
        setFrontendStatus("connected");
        setFrontendDetail(null);
      };

      socket.onmessage = (event: MessageEvent<string>) => {
        let parsed: ServerMessage | null = null;
        try {
          parsed = parseServerMessage(JSON.parse(event.data));
        } catch (error) {
          console.error("Failed to parse server message", error);
          setServerStatus("error");
          setServerDetail("Failed to parse server message.");
          return;
        }

        if (!parsed) {
          setServerStatus("error");
          setServerDetail("Received malformed message from server.");
          return;
        }

        switch (parsed.type) {
          case "image":
            if (parsed.channel === "pred") {
              setPredImageSrc(`data:image/jpeg;base64,${parsed.img}`);
            } else {
              setTrueImageSrc(`data:image/jpeg;base64,${parsed.img}`);
            }
            break;
          case "trainer_status":
            setTrainerState(parsed.status);
            break;
          case "connection_state":
            if (parsed.scope === "server") {
              setServerStatus(parsed.state);
              setServerDetail(parsed.detail ?? null);
              if (parsed.state === "disconnected" || parsed.state === "error") {
                setPredImageSrc("");
                setTrueImageSrc("");
                setTrainerState(null);
              }
            } else {
              setFrontendStatus(parsed.state);
              setFrontendDetail(parsed.detail ?? null);
            }
            break;
          case "error":
            setServerStatus("error");
            setServerDetail(parsed.detail ?? parsed.message);
            break;
        }
      };

      socket.onerror = () => {
        scheduleReconnect("WebSocket error encountered.");
      };

      socket.onclose = (event: CloseEvent) => {
        const detail =
          event.reason ||
          (event.wasClean ? "Connection closed." : "Connection lost.");
        scheduleReconnect(detail);
      };
    }

    setFrontendStatus("connecting");
    setFrontendDetail(null);
    connect();

    return () => {
      shouldReconnect = false;
      if (reconnectTimeout !== null) {
        window.clearTimeout(reconnectTimeout);
      }
      ws.current?.close();
      ws.current = null;
    };
  }, []);

  const handleSliderChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setIntervalValue(value);
    fetch("/control/slider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    });
  };

  const updateGroundTruthPreference = (enabled: boolean) => {
    fetch("/control/groundtruth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
  };

  const handleShowTrueToggle = () => {
    setShowTrue((current) => {
      const next = !current;
      updateGroundTruthPreference(next);
      return next;
    });
  };

  const statusClass = (status: ConnectionStatus | ConnectionStateKind) =>
    status === "connected"
      ? "text-green-600"
      : status === "error"
        ? "text-red-600"
        : status === "idle"
          ? "text-gray-600"
          : "text-amber-600";
  return (
    <div className="min-h-screen flex flex-col items-center justify-center space-y-2.5">
      <h1 className="text-4xl font-bold">LatentCSI Demo Viewer</h1>

      <div className="text-sm text-gray-600 space-y-1 text-center">
        <div>
          Frontend → Backend:{" "}
          <span className={`font-medium ${statusClass(frontendStatus)}`}>
            {frontendStatus}
          </span>
          {frontendDetail ? ` – ${frontendDetail}` : null}
        </div>
        <div>
          Backend → Server:{" "}
          <span className={`font-medium ${statusClass(serverStatus)}`}>
            {serverStatus}
          </span>
          {serverDetail ? ` – ${serverDetail}` : null}
        </div>
      </div>

      <div className="flex w-full max-w-5xl items-start justify-center gap-6">
        <div className="flex flex-1 justify-end">
          <StatusDisplay state={trainerState} />
        </div>
        <div className="flex shrink-0 flex-col items-center gap-4">
          <div className="flex items-center justify-center gap-4">
            <ImageDisplay
              imageSrc={predImageSrc}
              altText="Predicted"
              size={IMAGE_SIZE}
            />
            {showTrue && (
              <ImageDisplay
                imageSrc={trueImageSrc}
                altText="Ground truth"
                size={IMAGE_SIZE}
              />
            )}
          </div>
          <label className="inline-flex items-center space-x-2">
            <input
              type="checkbox"
              checked={showTrue}
              onChange={handleShowTrueToggle}
              className="peer sr-only"
            />
            <div className="relative h-6 w-11 rounded-full bg-gray-200 before:absolute before:left-[2px] before:top-[2px] before:h-5 before:w-5 before:rounded-full before:bg-white before:transition-transform peer-checked:bg-blue-600 peer-checked:before:translate-x-full" />
            <span className="text-sm font-medium">Show ground truth</span>
          </label>
          <ControlButtons />
          <IntervalSlider
            intervalValue={intervalValue}
            handleSliderChange={handleSliderChange}
          />
          <SDSettings />
        </div>
        <div className="flex flex-1 justify-start">
          <MessageMenu maxHeight={IMAGE_SIZE} />
        </div>
      </div>
    </div>
  );
}

export default App;
