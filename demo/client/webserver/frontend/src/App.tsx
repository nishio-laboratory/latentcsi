import { useState, useEffect, useRef } from "react";
import ImageDisplay from "./components/ImageDisplay";
import IntervalSlider from "./components/IntervalSlider";
import LearningRateControl from "./components/LearningRateControl";
import ControlButtons from "./components/ControlButtons";
import { SDSettings } from "./components/SDSettings";

function App() {
  const [predImageSrc, setPredImageSrc] = useState<string>("");
  const [trueImageSrc, setTrueImageSrc] = useState<string>("");
  const [intervalValue, setIntervalValue] = useState<number>(0.33);
  const [lrValue, setLrValue] = useState<string>("");
  const [showTrue, setShowTrue] = useState<boolean>(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws.current = new WebSocket(`${protocol}//${window.location.host}/ws`);
    ws.current.onmessage = (e: MessageEvent) => {
      const msg = JSON.parse(e.data);
      switch (msg.stream) {
        case "pred":
          setPredImageSrc(`data:image/jpeg;base64,${msg.img}`);
          break;
        case "true":
          setTrueImageSrc(`data:image/jpeg;base64,${msg.img}`);
          break;
      }
    };
    return () => {
      ws.current?.close();
    };
  }, []);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setIntervalValue(value);
    fetch("/control/slider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    });
  };

  const handleLrChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLrValue(e.target.value);
  };

  const handleLrSubmit = () => {
    fetch("/control/lr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value: parseFloat(lrValue) }),
    });
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-4xl font-bold">Live Viewer</h1>

      <div className="flex items-center space-x-4 mt-6">
        <ImageDisplay imageSrc={predImageSrc} altText="Predicted" size={256} />
        {showTrue && (
          <ImageDisplay
            imageSrc={trueImageSrc}
            altText="Ground truth"
            size={256}
          />
        )}
      </div>
      <div className="mt-10">
        <label className="inline-flex items-center space-x-2">
          <input
            type="checkbox"
            checked={showTrue}
            onChange={() => setShowTrue(s => !s)}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-gray-200 rounded-full peer-checked:bg-blue-600 relative before:content-[''] before:absolute before:top-[2px] before:left-[2px] before:bg-white before:w-5 before:h-5 before:rounded-full before:transition-transform peer-checked:before:translate-x-full" />
          <span className="text-sm font-medium">Show ground truth</span>
        </label>
      </div>

      <ControlButtons />

      <IntervalSlider
        intervalValue={intervalValue}
        handleSliderChange={handleSliderChange}
      />

      <LearningRateControl
        lrValue={lrValue}
        handleLrChange={handleLrChange}
        handleLrSubmit={handleLrSubmit}
      />

      <SDSettings />
    </div>
  );
}

export default App;
