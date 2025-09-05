import { useState } from 'react';

interface Img2ImgParams {
  enabled: boolean;
  prompt: string;
  negativePrompt: string;
  strength: number;
  cfg: number;
}

export function SDSettings() {
  const [params, setParams] = useState<Img2ImgParams>({
    enabled: false,
    prompt: '',
    negativePrompt: '',
    strength: 0.55,
    cfg: 7,
  });
  const [open, setOpen] = useState(false);

  const isOpen = open || params.enabled;

  const handleChange = <K extends keyof Img2ImgParams>(
    key: K,
    value: Img2ImgParams[K]
  ) => {
    const updated = { ...params, [key]: value };
    setParams(updated);
    if (key !== 'enabled') submit(updated);
  };

  const submit = async (payload: Img2ImgParams) => {
    console.log(JSON.stringify(payload));
    await fetch('/control/sdsettings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  };

  return (
    <div className="relative pt-2 px-2 w-full max-w-md border rounded-lg mt-5">
      <button
        aria-label={isOpen ? 'Close settings' : 'Open settings'}
        onClick={() => setOpen(o => !o)}
        className="absolute top-2 right-2 p-1 rounded hover:bg-gray-200"
      >
        {/* Arrow icon */}
        <svg
          className={`w-4 h-4 transform transition-transform duration-200 ${
            isOpen ? 'rotate-90' : 'rotate-0'
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 5l7 7-7 7"
          />
        </svg>
      </button>

      <label className="inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          className="sr-only peer"
          checked={params.enabled}
          onChange={e => {
            const enabled = e.target.checked;
            handleChange('enabled', enabled);
            if (enabled) submit({ ...params, enabled });
          }}
        />
        <div className="relative w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:bg-blue-600 dark:peer-checked:bg-blue-600 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:after:border-white" />
        <span className="ms-3 text-sm font-medium text-gray-900 dark:text-gray-300">
          Apply Stable Diffusion
        </span>
      </label>

      {isOpen && (
        <div className="mt-4 space-y-4">
          <div>
            <label className="block text-sm font-medium">Prompt</label>
            <input
              type="text"
              value={params.prompt}
              onChange={e => handleChange('prompt', e.target.value)}
              className="mt-1 block w-full border rounded p-2"
            />
          </div>

          <div>
            <label className="block text-sm font-medium">Negative Prompt</label>
            <input
              type="text"
              value={params.negativePrompt}
              onChange={e => handleChange('negativePrompt', e.target.value)}
              className="mt-1 block w-full border rounded p-2"
            />
          </div>

          <div>
            <label className="block text-sm font-medium">
              Strength: {params.strength.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={params.strength}
              onChange={e => handleChange('strength', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium">
              CFG: {params.cfg.toFixed(2)}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              step="0.1"
              value={params.cfg}
              onChange={e => handleChange('cfg', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      )}
    </div>
  );
}
