interface ImageDisplayProps {
  imageSrc: string;
  altText: string;
}

function ImageDisplay({ imageSrc, altText }: ImageDisplayProps) {
  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <img id="image" src={imageSrc} alt={altText} className="rounded-lg" width="512" height="512" />
    </div>
  );
}

export default ImageDisplay;
