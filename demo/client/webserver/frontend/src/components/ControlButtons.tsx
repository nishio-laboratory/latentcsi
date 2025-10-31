function ControlButtons() {
  const sendCommand = (endpoint: string): void => {
    fetch(endpoint, { method: "POST" });
  };
  return (
    <div className="flex">
      <button
        onClick={() => sendCommand("/control/start")}
        className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-l-md"
      >
        Start
      </button>
      <button
        onClick={() => sendCommand("/control/stop")}
        className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-r-md"
      >
        Stop
      </button>
    </div>
  );
}

export default ControlButtons;
