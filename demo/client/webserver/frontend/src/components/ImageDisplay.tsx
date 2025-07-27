interface ImageDisplayProps {
  imageSrc: string;
  altText: string;
  size?: number; // in pixels, default to 256
}

function ImageDisplay({ imageSrc, altText, size = 128 }: ImageDisplayProps) {
  const px = `${size}px`;
  return (
    <div
      className="rounded-lg border text-center mx-1 mt-5"
      style={{ width: px, height: px }}
    >
      <img
        src={imageSrc}
        alt={altText}
        className="rounded-lg"
        style={{ width: px, height: px, objectFit: "contain", lineHeight: px }}
      />
    </div>
  );
}

export default ImageDisplay;
