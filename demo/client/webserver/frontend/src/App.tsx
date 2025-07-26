import { useState, useEffect, useRef } from 'react';
import ImageDisplay from './components/ImageDisplay';
import IntervalSlider from './components/IntervalSlider';
import LearningRateControl from './components/LearningRateControl';
import ControlButtons from './components/ControlButtons';

function App() {
  const [predImageSrc, setPredImageSrc] = useState<string>('');
  const [trueImageSrc, setTrueImageSrc] = useState<string>('');
  const [intervalValue, setIntervalValue] = useState<number>(0.33);
  const [lrValue, setLrValue] = useState<string>('');
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8123/ws');
    ws.current.onmessage = (e: MessageEvent) => {
      const msg = JSON.parse(e.data);
      switch (msg.stream) {
        case "pred":
          console.log(msg);
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
    const value: number = parseFloat(e.target.value);
    setIntervalValue(value);
    fetch('/control/slider', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value }),
    });
  };

  const handleLrChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLrValue(e.target.value);
  };

  const handleLrSubmit = (): void => {
    fetch('/control/lr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: parseFloat(lrValue) }),
    });
  };


  return (
    <div className="bg-gray-900 min-h-screen text-white flex flex-col items-center justify-center">
      <h1 className="text-4xl font-bold mb-8">Live Viewer</h1>
      <ImageDisplay imageSrc={predImageSrc} altText="pred" />
      <ImageDisplay imageSrc={trueImageSrc} altText="true" />
      <IntervalSlider intervalValue={intervalValue} handleSliderChange={handleSliderChange} />
      <LearningRateControl lrValue={lrValue} handleLrChange={handleLrChange} handleLrSubmit={handleLrSubmit} />
      <ControlButtons/>
    </div>
  );
}

export default App;
