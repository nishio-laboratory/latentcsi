import { useState, useEffect, useRef } from 'react';

function App() {
  const [imageSrc, setImageSrc] = useState('');
  const [intervalValue, setIntervalValue] = useState(0.33);
  const [lrValue, setLrValue] = useState('');
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8123/ws');
    ws.current.onmessage = (e) => {
      setImageSrc(`data:image/jpeg;base64,${e.data}`);
    };
    return () => {
      ws.current.close();
    };
  }, []);

  const handleSliderChange = (e) => {
    const value = parseFloat(e.target.value);
    setIntervalValue(value);
    fetch('/control/slider', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value }),
    });
  };

  const handleLrChange = (e) => {
    setLrValue(e.target.value);
  };

  const handleLrSubmit = () => {
    fetch('/control/lr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: parseFloat(lrValue) }),
    });
  };

  const sendCommand = (endpoint) => {
    fetch(endpoint, { method: 'POST' });
  };

  return (
    <div className="bg-gray-900 min-h-screen text-white flex flex-col items-center justify-center">
      <h1 className="text-4xl font-bold mb-8">Live Viewer</h1>
      <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
        <img id="image" src={imageSrc} alt="Live feed" className="rounded-lg" />
      </div>
      <div className="w-full max-w-md mt-8">
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
        <div className="text-center mt-2">Update Interval: {intervalValue.toFixed(2)}s</div>
      </div>
      <div className="flex items-center mt-4">
        <input
          type="number"
          placeholder="Learning Rate"
          value={lrValue}
          onChange={handleLrChange}
          className="bg-gray-700 text-white px-4 py-2 rounded-l-md focus:outline-none"
        />
        <button
          onClick={handleLrSubmit}
          className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-r-md"
        >
          Set LR
        </button>
      </div>
      <div className="flex mt-8">
        <button
          onClick={() => sendCommand('/control/start')}
          className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-l-md"
        >
          Start
        </button>
        <button
          onClick={() => sendCommand('/control/stop')}
          className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-r-md"
        >
          Stop
        </button>
      </div>
    </div>
  );
}

export default App;