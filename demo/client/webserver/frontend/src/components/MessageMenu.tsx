import { useMemo, useState } from "react";

type VoidMessage =
  | "start_rec"
  | "stop_rec"
  | "start_train"
  | "stop_train"
  | "reset";

const MESSAGE_ENDPOINT = "/control/msg";

function postMessage(value: string) {
  fetch(MESSAGE_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
}

interface MessageMenuProps {
  maxHeight?: number;
}

export function MessageMenu({ maxHeight = 256 }: MessageMenuProps) {
  const [learningRateInput, setLearningRateInput] = useState("0.001");

  const parsedLearningRate = useMemo(
    () => Number(learningRateInput),
    [learningRateInput],
  );
  const learningRateIsValid = Number.isFinite(parsedLearningRate);

  const handleVoidMessage = (message: VoidMessage) => {
    postMessage(message);
  };

  const handleLearningRateSubmit = () => {
    if (!learningRateIsValid) {
      return;
    }
    postMessage(`set_lr(${parsedLearningRate})`);
  };

  return (
    <div
      className="flex flex-col gap-3 overflow-y-auto rounded border border-gray-200 bg-white p-4 shadow-sm"
      style={{ maxHeight }}
    >
      <h2 className="text-lg font-semibold text-gray-900">Server controls</h2>
      <div className="flex flex-col gap-2">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => handleVoidMessage("start_rec")}
            className="flex-1 rounded bg-blue-500 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-white transition hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
          >
            Start rec
          </button>
          <button
            type="button"
            onClick={() => handleVoidMessage("stop_rec")}
            className="flex-1 rounded bg-blue-500 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-white transition hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
          >
            Stop rec
          </button>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => handleVoidMessage("start_train")}
            className="flex-1 rounded bg-blue-500 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-white transition hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
          >
            Start train
          </button>
          <button
            type="button"
            onClick={() => handleVoidMessage("stop_train")}
            className="flex-1 rounded bg-blue-500 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-white transition hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
          >
            Stop train
          </button>
        </div>
        <button
          type="button"
          onClick={() => handleVoidMessage("reset")}
          className="rounded bg-red-600 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-white transition hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1"
        >
          Reset
        </button>
      </div>
      <div className="flex flex-col gap-2 border-t border-gray-200 pt-3">
        <label
          className="text-sm font-medium text-gray-800"
          htmlFor="learning-rate-input"
        >
          Learning rate
        </label>
        <div className="flex items-center gap-2">
          <input
            id="learning-rate-input"
            type="number"
            value={learningRateInput}
            onChange={(event) => setLearningRateInput(event.target.value)}
            placeholder="0.001"
            step="any"
            className="w-32 rounded border border-gray-300 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button
            type="button"
            onClick={handleLearningRateSubmit}
            disabled={!learningRateIsValid}
            className="rounded bg-indigo-500 px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:bg-indigo-300 disabled:text-indigo-100"
          >
            Set
          </button>
        </div>
        {!learningRateIsValid ? (
          <p className="text-xs text-red-600">
            Enter a valid number to update the learning rate.
          </p>
        ) : null}
      </div>
    </div>
  );
}

export default MessageMenu;
