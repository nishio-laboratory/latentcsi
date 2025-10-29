interface ImageDisplayProps {
  imageSrc: string;
  altText: string;
  size?: number; // in pixels, default to 256
}

function ImageDisplay({ imageSrc, altText, size = 128 }: ImageDisplayProps) {
  const px = `${size}px`;
  const hasImage = imageSrc.length > 0;

  return (
    <div className="mx-1 mt-5 flex flex-col items-center" style={{ width: px }}>
      <p className="mb-2 text-sm font-medium text-gray-700">{altText}</p>
      <div
        className="relative flex items-center justify-center rounded-lg border bg-gray-100 text-xs text-gray-500"
        style={{ width: px, height: px }}
        aria-live="polite"
      >
        {hasImage ? (
          <img
            src={imageSrc}
            alt={altText}
            className="h-full w-full rounded-lg object-contain"
          />
        ) : (
          <span>No image available</span>
        )}
      </div>
    </div>
  );
}

export default ImageDisplay;
