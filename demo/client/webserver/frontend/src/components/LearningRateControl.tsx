import React from "react";

interface LearningRateControlProps {
  lrValue: string;
  handleLrChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleLrSubmit: () => void;
}

function LearningRateControl({
  lrValue,
  handleLrChange,
  handleLrSubmit,
}: LearningRateControlProps) {
  return (
    <div className="flex items-center mt-4">
      <input
        type="number"
        placeholder="Learning Rate"
        value={lrValue}
        onChange={handleLrChange}
        className="bg-gray-300 rounded-l px-4 py-2 focus:outline-none"
      />
      <button
        onClick={handleLrSubmit}
        className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-r"
      >
        Set LR
      </button>
    </div>
  );
}

export default LearningRateControl;
