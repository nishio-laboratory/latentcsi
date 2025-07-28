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
    <div className="w-full max-w-md mt-5">
      <input
        type="range"
        id="slider"
        min="0.1"
        max="1"
        step="0.05"
        value={intervalValue}
        onInput={handleSliderChange}
        className="w-full"
      />
      <div className="text-center">
        Update Interval: {intervalValue.toFixed(2)}s
      </div>
    </div>
  );
}

export default IntervalSlider;
