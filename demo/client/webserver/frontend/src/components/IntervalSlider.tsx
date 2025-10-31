import React from "react";

interface IntervalSliderProps {
  intervalValue: number;
  handleSliderChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

function IntervalSlider({
  intervalValue,
  handleSliderChange,
}: IntervalSliderProps) {
  return (
    <div className="w-full max-w-xs">
      <div className="flex items-center gap-3">
        <input
          type="range"
          id="slider"
          min="0"
          max="1"
          step="0.05"
          value={intervalValue}
          onInput={handleSliderChange}
          className="w-full"
        />
        <span className="w-14 text-right text-sm font-medium text-gray-700">
          {intervalValue.toFixed(2)}s
        </span>
      </div>
    </div>
  );
}

export default IntervalSlider;
