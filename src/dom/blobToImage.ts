import { env } from "../env";

export function blobToImage(blb: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    if (!(blb instanceof Blob)) {
      return reject("blobToImage - expected blob to be of type: Blob");
    }

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        return reject("blobToImage - expected reader.result to be a string, in onload");
      }

      const img = env.getEnv().createImageElement();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(blb);
  });
}
