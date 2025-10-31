export type TrainerState = {
  started: boolean;
  batches_trained: number;
  reservoir_size: number;
  recording: boolean;
  training: boolean;
};

type StatusDisplayProps = {
  state: TrainerState | null;
};

function StatusDisplay({ state }: StatusDisplayProps) {
  return (
    <div className="w-60 flex-shrink-0 rounded border border-gray-300 bg-white/80 p-4 shadow">
      <h2 className="text-lg font-semibold text-gray-800">Trainer Status</h2>
      {!state ? (
        <p className="mt-2 text-sm text-gray-500">
          Waiting for trainer status…
        </p>
      ) : (
        <dl className="mt-2 space-y-1 text-sm text-gray-700">
          <div className="flex justify-between">
            <dt>Started:</dt>
            <dd className="font-medium">{state.started ? "Yes" : "No"}</dd>
          </div>
          <div className="flex justify-between">
            <dt>Training:</dt>
            <dd className="font-medium">{state.training ? "On" : "Off"}</dd>
          </div>
          <div className="flex justify-between">
            <dt>Recording:</dt>
            <dd className="font-medium">{state.recording ? "On" : "Off"}</dd>
          </div>
          <div className="flex justify-between">
            <dt>Batches trained:</dt>
            <dd className="font-medium">{state.batches_trained}</dd>
          </div>
          <div className="flex justify-between">
            <dt>Reservoir size:</dt>
            <dd className="font-medium">{state.reservoir_size}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}

export default StatusDisplay;
