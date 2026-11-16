import * as ort from "onnxruntime-web/wasm";

const MODEL_URL = `${import.meta.env.BASE_URL}models/yolo11n.onnx`;
const INPUT_SIZE = 640;
const CLASS_NAMES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
  "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

ort.env.wasm.numThreads = 1;

let sessionPromise;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      executionMode: "sequential",
    }).catch((error) => {
      sessionPromise = undefined;
      throw error;
    });
  }
  return sessionPromise;
}

function prepareInput(bitmap) {
  const scale = Math.min(INPUT_SIZE / bitmap.width, INPUT_SIZE / bitmap.height);
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const left = Math.round((INPUT_SIZE - width) / 2 - 0.1);
  const top = Math.round((INPUT_SIZE - height) / 2 - 0.1);
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(114, 114, 114)";
  context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  context.drawImage(bitmap, left, top, width, height);

  const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  const input = new Float32Array(3 * planeSize);
  for (let i = 0; i < planeSize; i += 1) {
    input[i] = pixels[i * 4] / 255;
    input[planeSize + i] = pixels[i * 4 + 1] / 255;
    input[planeSize * 2 + i] = pixels[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale,
    left,
    top,
  };
}

function intersectionOverUnion(a, b) {
  const width = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const height = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const intersection = width * height;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / Math.max(areaA + areaB - intersection, 1e-6);
}

function decodeOutput(output, threshold, imageWidth, imageHeight, scale, left, top) {
  const dims = output.dims;
  const channelsFirst = dims[1] < dims[2];
  const attributes = channelsFirst ? dims[1] : dims[2];
  const candidates = channelsFirst ? dims[2] : dims[1];
  const read = (attribute, candidate) => output.data[channelsFirst
    ? attribute * candidates + candidate
    : candidate * attributes + attribute];

  const boxes = [];
  for (let candidate = 0; candidate < candidates; candidate += 1) {
    let classId = -1;
    let confidence = threshold;
    for (let classIndex = 4; classIndex < attributes; classIndex += 1) {
      const score = read(classIndex, candidate);
      if (score > confidence) {
        confidence = score;
        classId = classIndex - 4;
      }
    }
    if (classId < 0 || classId >= CLASS_NAMES.length) continue;

    const centerX = read(0, candidate);
    const centerY = read(1, candidate);
    const boxWidth = read(2, candidate);
    const boxHeight = read(3, candidate);
    const x1 = Math.max(0, Math.min(imageWidth, (centerX - boxWidth / 2 - left) / scale));
    const y1 = Math.max(0, Math.min(imageHeight, (centerY - boxHeight / 2 - top) / scale));
    const x2 = Math.max(0, Math.min(imageWidth, (centerX + boxWidth / 2 - left) / scale));
    const y2 = Math.max(0, Math.min(imageHeight, (centerY + boxHeight / 2 - top) / scale));
    if (x2 <= x1 || y2 <= y1) continue;
    boxes.push({ classId, confidence, box: { x1, y1, x2, y2 } });
  }

  boxes.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const candidate of boxes) {
    if (kept.some((selected) => selected.classId === candidate.classId && intersectionOverUnion(selected.box, candidate.box) > 0.45)) continue;
    kept.push(candidate);
    if (kept.length === 300) break;
  }

  return kept.map((item, index) => ({
    id: index + 1,
    class_id: item.classId,
    label: CLASS_NAMES[item.classId],
    confidence: Number(item.confidence.toFixed(4)),
    box: Object.fromEntries(Object.entries(item.box).map(([key, value]) => [key, Number(value.toFixed(2))])),
  }));
}

export async function detectObjects(file, threshold) {
  const bitmap = await createImageBitmap(file);
  try {
    const { tensor, scale, left, top } = prepareInput(bitmap);
    const session = await getSession();
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const detections = decodeOutput(
      outputs[session.outputNames[0]], threshold, bitmap.width, bitmap.height, scale, left, top,
    );
    return {
      filename: file.name,
      image: { width: bitmap.width, height: bitmap.height },
      count: detections.length,
      detections,
    };
  } finally {
    bitmap.close();
  }
}
