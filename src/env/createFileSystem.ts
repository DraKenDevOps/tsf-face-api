import { FileSystem } from "./types";

export function createFileSystem(fs?: any): FileSystem {
  let requireFsError = "";

  if (!fs) {
    try {
      if (typeof require === "function") {
        fs = require("fs");
      }
    } catch (err: any) {
      requireFsError = err ? err.toString() : "unknown error";
    }
  }

  const readFile = fs
    ? function (filePath: string) {
        return new Promise<Buffer>((res, rej) => {
          fs.readFile(filePath, function (err: any, buffer: Buffer) {
            return err ? rej(err) : res(buffer);
          });
        });
      }
    : async function (filePath: string) {
        try {
          const nodeFs = await import("fs");
          return await nodeFs.promises.readFile(filePath);
        } catch (err: any) {
          throw new Error(`readFile - failed to require fs in nodejs environment with error: ${requireFsError || err}`);
        }
      };

  return {
    readFile
  };
}
