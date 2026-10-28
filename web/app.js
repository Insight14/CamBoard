/**
 * CamBoard — Main Frontend Application Logic
 * Integrates ONNX Runtime Web, MediaPipe Hands, Multi-Stroke Context-Aware Math Recognition,
 * 3-Second Hand-Off Auto-Finish, and Fist-Grab Draggable Cards.
 */

// Global State
const state = {
  models: {
    gesture: null,
    fingerspelling: null,
    stroke_shapes: null,
    stroke_chars: null,
  },
  cameraActive: false,
  hands: null,
  airDrawing: true,
  isAirDrawing: false,
  autocorrect: true,

  // Jitter-free exponential smoothing coordinates
  smoothHandX: null,
  smoothHandY: null,
  lastAirPos: null,

  // Multi-stroke symbol buffer & debounce
  currentSymbolStrokes: [],
  symbolDebounceTimer: null,

  // Running Equation & Context
  equationTokens: [], // Array of { text, type: 'number'|'operator'|'variable'|'shape', bbox }
  
  // Hand-off-screen 3-second auto-finish timer
  handOffStartTime: null,
  autoFinishCancelled: false,

  // Saved Equation Cards & Fist Drag Interaction
  savedCards: [],
  draggedCard: null,
  dragOffset: { x: 0, y: 0 },
  isMouseDragging: false,

  recordedSamples: [],
  lastLandmarks: null,
  strokeHistory: [], // Undo stack

  fps: 0,
  frameCount: 0,
  lastFpsUpdate: performance.now(),
};

// Class name definitions
const GESTURE_CLASSES = ["pinch", "palm", "fist", "point", "swipe_start", "thumbs_up", "palm_out", "none"];
const ASL_CLASSES = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"];
const SHAPE_CLASSES = ["circle", "square", "triangle", "star", "diamond", "arrow", "line", "rectangle", "hexagon", "zigzag"];

// DOM Elements
const webcamVideo = document.getElementById("webcam-video");
const cameraOverlay = document.getElementById("camera-overlay");
const overlayCtx = cameraOverlay.getContext("2d");
const whiteboardCanvas = document.getElementById("whiteboard-canvas");
const canvasCtx = whiteboardCanvas.getContext("2d");
const airCursor = document.getElementById("air-cursor");
const cardsLayer = document.getElementById("cards-layer");
const autofinishToast = document.getElementById("autofinish-toast");
const countdownVal = document.getElementById("countdown-val");
const btnCancelFinish = document.getElementById("btn-cancel-finish");

// Status indicators
const ortStatus = document.getElementById("ort-status");
const cameraStatus = document.getElementById("camera-status");
const pillCamera = document.getElementById("pill-camera");
const pillOrt = document.getElementById("pill-ort");
const fpsCounter = document.getElementById("fps-counter");

// Telemetry & Results
const hudGesture = document.getElementById("hud-gesture");
const hudAction = document.getElementById("hud-action");
const telGesture = document.getElementById("tel-gesture");
const telConfidence = document.getElementById("tel-confidence");
const telLatency = document.getElementById("tel-latency");
const meterFill = document.getElementById("meter-fill");
const resEquation = document.getElementById("res-equation");
const resShape = document.getElementById("res-shape");
const resShapeConf = document.getElementById("res-shape-conf");
const resChar = document.getElementById("res-char");
const resCharSub = document.getElementById("res-char-sub");
const resDisambiguation = document.getElementById("res-disambiguation");
const resCardsCount = document.getElementById("res-cards-count");
const btnAutocorrect = document.getElementById("btn-autocorrect");
const btnUndo = document.getElementById("btn-undo");
const btnFinishEq = document.getElementById("btn-finish-eq");

// Active Drawing Strokes
let activeStroke = [];
let allStrokes = [];

// Initialize ONNX Runtime Models
async function initONNXModels() {
  ortStatus.textContent = "Loading WebAssembly...";
  try {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

    console.log("Loading gesture.onnx...");
    state.models.gesture = await ort.InferenceSession.create("models/gesture.onnx");

    try {
      state.models.stroke_shapes = await ort.InferenceSession.create("models/stroke_shapes.onnx");
      state.models.fingerspelling = await ort.InferenceSession.create("models/fingerspelling.onnx");
      state.models.stroke_chars = await ort.InferenceSession.create("models/stroke_chars.onnx");
    } catch (e) {
      console.warn("Optional auxiliary models note:", e);
    }

    ortStatus.textContent = "Ready (Wasm)";
    pillOrt.classList.add("active");
    console.log("ONNX Runtime Web initialized successfully.");
  } catch (err) {
    console.error("Error loading ONNX models:", err);
    ortStatus.textContent = "Ready (Geometric)";
    pillOrt.classList.add("active");
  }
}

// MediaPipe Hands Initialization
function initMediaPipe() {
  state.hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
  });

  state.hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });

  state.hands.onResults(onHandResults);
}

// Webcam start/stop
async function toggleCamera() {
  const btn = document.getElementById("btn-toggle-cam");
  if (!state.cameraActive) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      webcamVideo.srcObject = stream;
      await webcamVideo.play();

      cameraOverlay.width = webcamVideo.videoWidth || 640;
      cameraOverlay.height = webcamVideo.videoHeight || 480;

      state.cameraActive = true;
      cameraStatus.textContent = "Active";
      pillCamera.classList.add("active");
      btn.textContent = "Stop Camera";
      btn.className = "btn btn-danger";

      let isBusy = false;
      let lastSend = 0;

      async function processVideoFrame() {
        if (!state.cameraActive) return;

        const now = performance.now();
        // Self-healing watchdog: unlock if hung for > 350ms
        if (isBusy && (now - lastSend > 350)) {
          isBusy = false;
        }

        if (!isBusy && (now - lastSend >= 25) && webcamVideo.readyState >= 2 && webcamVideo.videoWidth > 0) {
          isBusy = true;
          lastSend = now;
          try {
            await state.hands.send({ image: webcamVideo });
          } catch (err) {
            console.warn("Hands detection frame error:", err);
          } finally {
            isBusy = false;
          }
        }

        if (state.cameraActive) {
          requestAnimationFrame(processVideoFrame);
        }
      }

      requestAnimationFrame(processVideoFrame);
    } catch (err) {
      alert("Could not access webcam: " + err.message);
    }
  } else {
    const stream = webcamVideo.srcObject;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    webcamVideo.srcObject = null;
    state.cameraActive = false;
    state.smoothHandX = null;
    state.smoothHandY = null;
    cameraStatus.textContent = "Off";
    pillCamera.classList.remove("active");
    btn.textContent = "Start Camera";
    btn.className = "btn btn-primary";
    overlayCtx.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);
    airCursor.style.display = "none";
    hudGesture.textContent = "Gesture: Detecting...";
    hudAction.textContent = "Mode: Waiting";
  }
}

// MediaPipe Results Callback
async function onHandResults(results) {
  state.frameCount++;
  const now = performance.now();
  if (now - state.lastFpsUpdate >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFpsUpdate));
    fpsCounter.textContent = `${state.fps} FPS`;
    state.frameCount = 0;
    state.lastFpsUpdate = now;
  }

  overlayCtx.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);

  // 1. Check if NO HAND is visible on screen
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    hudGesture.textContent = "Gesture: No Hand";
    hudAction.textContent = "Mode: Waiting";
    telGesture.textContent = "None";
    telConfidence.textContent = "0%";
    meterFill.style.width = "0%";
    airCursor.style.display = "none";

    // If currently air drawing, end the stroke cleanly
    if (state.isAirDrawing) endAirStroke();

    // Check 3-SECOND HAND-OFF-SCREEN AUTO-FINISH TRIGGER
    handleHandOffScreenTimer(now);
    return;
  }

  // Hand is on screen: cancel auto-finish countdown
  state.handOffStartTime = null;
  state.autoFinishCancelled = false;
  autofinishToast.classList.remove("visible");

  // Pick foreground hand with largest palm
  let landmarks = results.multiHandLandmarks[0];
  if (results.multiHandLandmarks.length > 1) {
    let maxPalm = -1;
    for (const h of results.multiHandLandmarks) {
      const pSize = Math.hypot(h[9].x - h[0].x, h[9].y - h[0].y);
      if (pSize > maxPalm) {
        maxPalm = pSize;
        landmarks = h;
      }
    }
  }
  state.lastLandmarks = landmarks;

  // Draw hand skeleton on camera overlay
  drawHandSkeleton(overlayCtx, landmarks, cameraOverlay.width, cameraOverlay.height);

  // Palm reference scale
  const dxPalm = landmarks[9].x - landmarks[0].x;
  const dyPalm = landmarks[9].y - landmarks[0].y;
  const dzPalm = (landmarks[9].z || 0) - (landmarks[0].z || 0);
  const palmScale = Math.hypot(dxPalm, dyPalm, dzPalm);

  // Pinch contact distance between Thumb tip (4) and Index tip (8)
  const dxTip = landmarks[4].x - landmarks[8].x;
  const dyTip = landmarks[4].y - landmarks[8].y;
  const dzTip = (landmarks[4].z || 0) - (landmarks[8].z || 0);
  const tip3DDist = Math.hypot(dxTip, dyTip, dzTip);
  const normalizedPinchDist = tip3DDist / Math.max(palmScale, 1e-4);

  // Strict physical touch criteria with drawing hysteresis:
  // Starts drawing ONLY when thumb and index finger completely touch (< 0.16)
  // Keeps drawing during continuous strokes until fingers clearly separate (> 0.26)
  const PINCH_TOUCH_THRESHOLD = 0.16;
  const PINCH_RELEASE_THRESHOLD = 0.26;

  const isPhysicallyTouching = state.isAirDrawing
    ? normalizedPinchDist < PINCH_RELEASE_THRESHOLD
    : normalizedPinchDist < PINCH_TOUCH_THRESHOLD;

  // Rotation-invariant finger extension / curl detection relative to wrist (0)
  function isFingerExt(tipIdx, pipIdx) {
    const dTip = Math.hypot(landmarks[tipIdx].x - landmarks[0].x, landmarks[tipIdx].y - landmarks[0].y);
    const dPip = Math.hypot(landmarks[pipIdx].x - landmarks[0].x, landmarks[pipIdx].y - landmarks[0].y);
    return dTip > dPip * 1.15;
  }
  function isFingerCurled(tipIdx, pipIdx) {
    const dTip = Math.hypot(landmarks[tipIdx].x - landmarks[0].x, landmarks[tipIdx].y - landmarks[0].y);
    const dPip = Math.hypot(landmarks[pipIdx].x - landmarks[0].x, landmarks[pipIdx].y - landmarks[0].y);
    return dTip < dPip * 1.05;
  }

  const indexExt = isFingerExt(8, 6);
  const middleExt = isFingerExt(12, 10);
  const ringExt = isFingerExt(16, 14);
  const pinkyExt = isFingerExt(20, 18);

  const indexCurled = isFingerCurled(8, 6);
  const middleCurled = isFingerCurled(12, 10);
  const ringCurled = isFingerCurled(16, 14);
  const pinkyCurled = isFingerCurled(20, 18);

  // Clenched fist: all 4 fingers tightly curled into palm base
  const isFist = indexCurled && middleCurled && ringCurled && pinkyCurled;

  // Determine active gesture
  let detectedGesture = "palm";
  let confidence = 0.90;
  const t0 = performance.now();

  if (isFist) {
    detectedGesture = "fist";
    confidence = 0.96;
  } else if (isPhysicallyTouching) {
    detectedGesture = "pinch";
    confidence = Math.min(0.99, 0.75 + (PINCH_RELEASE_THRESHOLD - normalizedPinchDist) * 2.0);
  } else if (indexExt && middleCurled && ringCurled && pinkyCurled) {
    detectedGesture = "point";
    confidence = 0.94;
  } else if (indexCurled && middleCurled && ringCurled && pinkyCurled && landmarks[4].y < landmarks[3].y) {
    detectedGesture = "thumbs_up";
    confidence = 0.92;
  } else {
    detectedGesture = "palm";
    confidence = 0.95;
  }

  const latencyMs = (performance.now() - t0).toFixed(1);

  // Update UI Telemetry & Badges
  hudGesture.textContent = `Gesture: ${detectedGesture.toUpperCase()}`;
  telGesture.textContent = detectedGesture;
  telConfidence.textContent = `${Math.round(confidence * 100)}%`;
  meterFill.style.width = `${Math.round(confidence * 100)}%`;
  telLatency.textContent = `${latencyMs} ms`;

  // Update Matrix Items
  document.querySelectorAll(".matrix-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.gesture === detectedGesture);
  });

  // Handle Air-Drawing & Fist-Grab Card Manipulation
  handleAirInteraction(landmarks, isPhysicallyTouching, isFist);
}

// 3-Second Hand-Off-Screen Auto-Finish Handler
function handleHandOffScreenTimer(now) {
  // Only trigger if there are active strokes or an active equation drawn on the whiteboard
  if (allStrokes.length === 0 && state.equationTokens.length === 0) {
    autofinishToast.classList.remove("visible");
    state.handOffStartTime = null;
    return;
  }

  if (state.autoFinishCancelled) return;

  if (state.handOffStartTime === null) {
    state.handOffStartTime = now;
  }

  const elapsed = now - state.handOffStartTime;
  const remaining = Math.max(0, 3.0 - (elapsed / 1000));
  countdownVal.textContent = `${remaining.toFixed(1)}s`;
  autofinishToast.classList.add("visible");

  if (elapsed >= 3000) {
    // 3 Seconds complete! Finalize and auto-fit equation into corner!
    autofinishToast.classList.remove("visible");
    state.handOffStartTime = null;
    finishCurrentEquationAndSaveCard();
  }
}

// Draw hand landmarks and connectors
function drawHandSkeleton(ctx, landmarks, w, h) {
  ctx.strokeStyle = "#00d2aa";
  ctx.fillStyle = "#4f8cff";
  ctx.lineWidth = 2;

  const connections = [
    [0,1],[1,2],[2,3],[3,4], // Thumb
    [0,5],[5,6],[6,7],[7,8], // Index
    [0,9],[9,10],[10,11],[11,12], // Middle
    [0,13],[13,14],[14,15],[15,16], // Ring
    [0,17],[17,18],[18,19],[19,20], // Pinky
    [5,9],[9,13],[13,17] // Palm MCPs
  ];

  for (const [i, j] of connections) {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    ctx.beginPath();
    ctx.moveTo(p1.x * w, p1.y * h);
    ctx.lineTo(p2.x * w, p2.y * h);
    ctx.stroke();
  }

  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Air Interaction (Drawing, Hover, and Fist Grab & Drag)
function handleAirInteraction(landmarks, isPhysicallyTouching, isFist) {
  if (!state.airDrawing) return;

  // Midpoint between thumb tip and index tip when touching, or index fingertip when hovering
  const targetPt = isPhysicallyTouching
    ? { x: (landmarks[4].x + landmarks[8].x) * 0.5, y: (landmarks[4].y + landmarks[8].y) * 0.5 }
    : { x: landmarks[8].x, y: landmarks[8].y };

  // Mirror camera X so hand motion matches user orientation
  const camX = 1.0 - targetPt.x;
  const camY = targetPt.y;

  // Smoothing filter
  if (state.smoothHandX === null) {
    state.smoothHandX = camX;
    state.smoothHandY = camY;
  } else {
    state.smoothHandX = state.smoothHandX * 0.25 + camX * 0.75;
    state.smoothHandY = state.smoothHandY * 0.25 + camY * 0.75;
  }

  // Normalized reach margins
  const minReachX = 0.12, maxReachX = 0.88;
  const minReachY = 0.10, maxReachY = 0.88;
  const u = Math.min(Math.max((state.smoothHandX - minReachX) / (maxReachX - minReachX), 0), 1);
  const v = Math.min(Math.max((state.smoothHandY - minReachY) / (maxReachY - minReachY), 0), 1);

  // Exact 1:1 canvas screen & coordinate alignment
  const canvasRect = whiteboardCanvas.getBoundingClientRect();
  const screenX = canvasRect.left + u * canvasRect.width;
  const screenY = canvasRect.top + v * canvasRect.height;

  airCursor.style.display = "block";
  airCursor.style.left = `${screenX}px`;
  airCursor.style.top = `${screenY}px`;

  const canvasX = u * whiteboardCanvas.width;
  const canvasY = v * whiteboardCanvas.height;

  // --- FIST GRAB & DRAG INTERACTION ---
  if (isFist) {
    hudAction.textContent = "Mode: GRAB (Fist)";
    hudAction.className = "badge badge-info";
    airCursor.className = "air-cursor fist";

    if (state.isAirDrawing) endAirStroke();

    // Check if cursor is over any equation card to grab
    if (!state.draggedCard) {
      for (const card of state.savedCards) {
        const cardRect = card.element.getBoundingClientRect();
        if (
          screenX >= cardRect.left - 10 &&
          screenX <= cardRect.right + 10 &&
          screenY >= cardRect.top - 10 &&
          screenY <= cardRect.bottom + 10
        ) {
          state.draggedCard = card;
          state.dragOffset = {
            x: screenX - cardRect.left,
            y: screenY - cardRect.top,
          };
          card.element.classList.add("grabbed");
          break;
        }
      }
    } else {
      // Drag active card smoothly across whiteboard
      const newLeft = screenX - canvasRect.left - state.dragOffset.x;
      const newTop = screenY - canvasRect.top - state.dragOffset.y;
      state.draggedCard.x = Math.max(0, Math.min(newLeft, canvasRect.width - 150));
      state.draggedCard.y = Math.max(0, Math.min(newTop, canvasRect.height - 80));
      state.draggedCard.element.style.left = `${state.draggedCard.x}px`;
      state.draggedCard.element.style.top = `${state.draggedCard.y}px`;
    }
    return;
  } else {
    // Release grabbed card when fist opens
    if (state.draggedCard) {
      state.draggedCard.element.classList.remove("grabbed");
      state.draggedCard = null;
    }
    airCursor.classList.remove("fist");
  }

  // --- DRAWING VS HOVER MODE ---
  if (isPhysicallyTouching) {
    hudAction.textContent = "Mode: DRAWING (Pinching)";
    hudAction.className = "badge badge-danger";
    airCursor.className = "air-cursor drawing";

    // Cancel pending debounce symbol timer while actively pinching new strokes
    if (state.symbolDebounceTimer) {
      clearTimeout(state.symbolDebounceTimer);
      state.symbolDebounceTimer = null;
    }

    if (!state.isAirDrawing) {
      state.isAirDrawing = true;
      activeStroke = [{ x: canvasX, y: canvasY }];
      state.lastAirPos = { x: canvasX, y: canvasY };
    } else {
      canvasCtx.beginPath();
      canvasCtx.moveTo(state.lastAirPos.x, state.lastAirPos.y);
      canvasCtx.lineTo(canvasX, canvasY);
      canvasCtx.strokeStyle = document.getElementById("pen-color").value;
      canvasCtx.lineWidth = parseInt(document.getElementById("pen-size").value, 10);
      canvasCtx.lineCap = "round";
      canvasCtx.lineJoin = "round";
      canvasCtx.stroke();

      activeStroke.push({ x: canvasX, y: canvasY });
      state.lastAirPos = { x: canvasX, y: canvasY };
    }
  } else {
    // Hover Mode: fingers unpinched
    hudAction.textContent = "Mode: Hover";
    hudAction.className = "badge badge-info";
    airCursor.className = "air-cursor";

    if (state.isAirDrawing) {
      endAirStroke();
    }
  }
}

// End individual air stroke segment (allows multi-stroke symbol completion)
function endAirStroke() {
  state.isAirDrawing = false;
  if (activeStroke.length >= 4) {
    saveCanvasHistory();
    allStrokes.push([...activeStroke]);
    state.currentSymbolStrokes.push([...activeStroke]);

    // Redraw canvas with all current strokes visible
    redrawAllStrokes();

    // Start symbol completion debounce (1000ms pause finalizes the multi-stroke symbol)
    if (state.symbolDebounceTimer) clearTimeout(state.symbolDebounceTimer);
    state.symbolDebounceTimer = setTimeout(() => {
      finalizeCurrentSymbol();
    }, 1000);
  }
  activeStroke = [];
}

// Finalize Multi-Stroke Symbol with Context Disambiguation
function finalizeCurrentSymbol() {
  if (state.currentSymbolStrokes.length === 0) return;

  const strokesToProcess = [...state.currentSymbolStrokes];
  state.currentSymbolStrokes = [];

  // Run Context-Aware Recognition on the collected strokes
  const result = recognizeSymbolWithContext(strokesToProcess, state.equationTokens);

  // Update equation tokens
  state.equationTokens.push(result);

  // Update Results Dashboard
  updateDashboardResults(result);

  // Autocorrect the symbol visually on canvas if enabled
  if (state.autocorrect) {
    applyAutocorrectToLastSymbol(result, strokesToProcess);
  }
}

// Context-Aware Recognition & Disambiguation Engine
function recognizeSymbolWithContext(strokes, previousTokens) {
  const nStrokes = strokes.length;
  const flatPoints = strokes.flat();
  const resampled = resampleStroke(flatPoints, 64);

  // Bounding box & Aspect Ratio
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of flatPoints) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const width = Math.max(maxX - minX, 1);
  const height = Math.max(maxY - minY, 1);
  const aspect = width / height;
  const diagonal = Math.hypot(width, height);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  // Context Analysis from previous tokens
  const prevToken = previousTokens.length > 0 ? previousTokens[previousTokens.length - 1] : null;
  const isAfterNumber = prevToken && (prevToken.type === "number" || prevToken.type === "variable");
  const isAfterOperator = prevToken && prevToken.type === "operator";

  let recognized = null;
  let disambiguationReason = "";

  // CASE A: Two-Stroke Symbols (+, ×, =, X, T, 4)
  if (nStrokes === 2) {
    const s1 = strokes[0];
    const s2 = strokes[1];
    const bbox1 = getStrokeBBox(s1);
    const bbox2 = getStrokeBBox(s2);

    const isS1Horiz = bbox1.width > bbox1.height * 1.5;
    const isS2Horiz = bbox2.width > bbox2.height * 1.5;
    const isS1Vert = bbox1.height > bbox1.width * 1.5;
    const isS2Vert = bbox2.height > bbox2.width * 1.5;

    // Check Equals Sign (=): Two parallel horizontal strokes
    if (isS1Horiz && isS2Horiz) {
      const vertDist = Math.abs(bbox1.cy - bbox2.cy);
      if (vertDist < height * 0.8) {
        return {
          text: "=",
          name: "Equals",
          type: "operator",
          confidence: 0.98,
          bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
        };
      }
    }

    // Check Plus (+) vs Letter 't' (One horizontal, one vertical intersecting)
    if ((isS1Horiz && isS2Vert) || (isS1Vert && isS2Horiz)) {
      // Disambiguation between '+' and 't':
      if (isAfterNumber || previousTokens.length === 0) {
        disambiguationReason = "Disambiguated to '+' (not 't') due to math/equation context";
        return {
          text: "+",
          name: "Plus Operator",
          type: "operator",
          confidence: 0.96,
          disambiguation: disambiguationReason,
          bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
        };
      } else {
        return {
          text: "t",
          name: "Letter t",
          type: "variable",
          confidence: 0.90,
          bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
        };
      }
    }

    // Check Multiplication (×) vs Variable 'x' (Two diagonal intersecting strokes)
    const s1Diag = Math.abs(bbox1.aspect - 1.0) < 0.8;
    const s2Diag = Math.abs(bbox2.aspect - 1.0) < 0.8;
    if (s1Diag && s2Diag) {
      if (isAfterNumber) {
        disambiguationReason = "Disambiguated to '×' (Multiply) between numbers";
        return {
          text: "×",
          name: "Multiplication",
          type: "operator",
          confidence: 0.95,
          disambiguation: disambiguationReason,
          bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
        };
      } else {
        return {
          text: "x",
          name: "Variable x",
          type: "variable",
          confidence: 0.94,
          bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
        };
      }
    }

    // Two-stroke Star or Angle
    const starGeom = checkStarGeometry(flatPoints, cx, cy, width, height);
    if (starGeom.isStar) {
      return {
        text: "⭐",
        name: "Star",
        type: "shape",
        confidence: 0.94,
        bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
      };
    }
  }

  // CASE B: Single Continuous Stroke (Shapes, Digits, Minus, Slash, Caret)
  const geom = analyzeSingleOrContinuousShape(flatPoints, cx, cy, width, height, diagonal);

  // Context Disambiguation for Single Stroke:
  if (geom.name === "circle") {
    // 0 (Zero) vs O (Letter) vs Circle shape
    if (isAfterNumber || isAfterOperator) {
      disambiguationReason = "Disambiguated to digit '0' in numerical expression";
      return {
        text: "0",
        name: "Digit 0",
        type: "number",
        confidence: 0.96,
        disambiguation: disambiguationReason,
        bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
      };
    } else if (width > 80 && height > 80) {
      return {
        text: "○",
        name: "Circle",
        type: "shape",
        confidence: 0.97,
        bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
      };
    } else {
      return {
        text: "0",
        name: "Digit 0 / O",
        type: "number",
        confidence: 0.92,
        bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
      };
    }
  }

  if (geom.name === "line") {
    // Minus (-) vs Division (/) vs 1 (One) vs Line shape
    if (aspect > 2.2) {
      disambiguationReason = "Disambiguated to '-' (Minus) operator";
      return {
        text: "-",
        name: "Minus / Subtraction",
        type: "operator",
        confidence: 0.96,
        disambiguation: disambiguationReason,
        bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
      };
    } else if (aspect < 0.4) {
      return {
        text: "1",
        name: "Digit 1",
        type: "number",
        confidence: 0.95,
        bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
      };
    } else {
      return {
        text: "/",
        name: "Division / Slash",
        type: "operator",
        confidence: 0.91,
        bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
      };
    }
  }

  // Geometric shapes: Star, Triangle, Square, Arrow, Diamond
  return {
    text: geom.symbol || geom.name.toUpperCase(),
    name: geom.name.toUpperCase(),
    type: geom.type || "shape",
    confidence: geom.confidence || 0.92,
    bbox: { minX, maxX, minY, maxY, cx, cy, width, height },
  };
}

// Helper: Bounding box of single stroke
function getStrokeBBox(stroke) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of stroke) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const width = Math.max(maxX - minX, 1);
  const height = Math.max(maxY - minY, 1);
  return {
    width,
    height,
    aspect: width / height,
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
  };
}

// Geometry intersection helpers for robust self-intersection detection
function segmentsIntersect(p1, p2, p3, p4) {
  function ccw(a, b, c) {
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x);
  }
  return ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4);
}

function countStrokeSelfIntersections(pts) {
  if (pts.length < 8) return 0;
  const sampled = resampleStroke(pts, 36);
  let count = 0;
  for (let i = 0; i < sampled.length - 2; i++) {
    for (let j = i + 2; j < sampled.length - 1; j++) {
      if (i === 0 && j >= sampled.length - 2) continue; // ignore closure point
      if (segmentsIntersect(sampled[i], sampled[i + 1], sampled[j], sampled[j + 1])) {
        count++;
      }
    }
  }
  return count;
}

// Douglas-Peucker Polygon Simplification for corner extraction
function simplifyPolygonDP(pts, epsilon) {
  if (pts.length <= 2) return [...pts];
  let maxDist = 0;
  let index = 0;
  const start = pts[0];
  const end = pts[pts.length - 1];
  const lineLen = Math.hypot(end.x - start.x, end.y - start.y);

  for (let i = 1; i < pts.length - 1; i++) {
    let dist = 0;
    if (lineLen < 1e-4) {
      dist = Math.hypot(pts[i].x - start.x, pts[i].y - start.y);
    } else {
      dist = Math.abs((end.y - start.y) * pts[i].x - (end.x - start.x) * pts[i].y + end.x * start.y - end.y * start.x) / lineLen;
    }
    if (dist > maxDist) {
      maxDist = dist;
      index = i;
    }
  }

  if (maxDist > epsilon) {
    const r1 = simplifyPolygonDP(pts.slice(0, index + 1), epsilon);
    const r2 = simplifyPolygonDP(pts.slice(index), epsilon);
    return r1.slice(0, r1.length - 1).concat(r2);
  } else {
    return [start, end];
  }
}

// Geometric Shape Analysis
function analyzeSingleOrContinuousShape(pts, cx, cy, width, height, diagonal) {
  const n = pts.length;
  const resampled = resampleStroke(pts, 64);
  const intersections = countStrokeSelfIntersections(pts);

  // Closure calculation
  const startPt = pts[0];
  const endPt = pts[n - 1];
  const closureDist = Math.hypot(startPt.x - endPt.x, startPt.y - endPt.y);
  let totalLength = 0;
  for (let i = 1; i < pts.length; i++) {
    totalLength += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  }
  const isClosed = (closureDist / diagonal < 0.38) || (closureDist / totalLength < 0.28);

  // Radial distances & variation
  const radii = resampled.map(p => Math.hypot(p.x - cx, p.y - cy));
  const meanR = radii.reduce((a, b) => a + b, 0) / 64;
  let varR = 0;
  for (const r of radii) varR += (r - meanR) * (r - meanR);
  const cvR = Math.sqrt(varR / 64) / Math.max(meanR, 1e-4);

  // Area & Circularity
  let areaSum = 0;
  for (let i = 0; i < 63; i++) {
    areaSum += resampled[i].x * resampled[i + 1].y - resampled[i + 1].x * resampled[i].y;
  }
  areaSum += resampled[63].x * resampled[0].y - resampled[0].x * resampled[63].y;
  const area = Math.abs(areaSum) * 0.5;
  const circularity = (4 * Math.PI * area) / Math.max(totalLength * totalLength, 1e-4);
  const boxFill = area / (width * height);

  // Count radial peaks and valleys
  let radialPeaks = 0;
  let radialValleys = 0;
  for (let i = 2; i < 62; i++) {
    if (radii[i] > radii[i - 1] && radii[i] > radii[i - 2] && radii[i] > radii[i + 1] && radii[i] > radii[i + 2]) {
      if (radii[i] > meanR * 1.10) radialPeaks++;
    }
    if (radii[i] < radii[i - 1] && radii[i] < radii[i - 2] && radii[i] < radii[i + 1] && radii[i] < radii[i + 2]) {
      if (radii[i] < meanR * 0.90) radialValleys++;
    }
  }

  // 1. STAR (Check self-intersections first — 5-point star always has 3-5 crossings!)
  if (intersections >= 3 || (radialPeaks >= 4 && (radialValleys >= 3 || cvR > 0.18))) {
    return { name: "star", symbol: "⭐", type: "shape", confidence: 0.98 };
  }

  // 2. CIRCLE / OVAL (Smooth loop, zero intersections, low radial variance)
  if (isClosed && intersections <= 1 && cvR < 0.22 && circularity > 0.46) {
    return { name: "circle", symbol: "○", type: "shape", confidence: 0.96 };
  }

  // Corner simplification
  const corners = simplifyPolygonDP(resampled, 0.08 * diagonal);
  const numCorners = corners.length - 1; // excluding closure endpoint

  // 3. TRIANGLE (Zero intersections, 3 prominent corners, moderate fill)
  if (isClosed && intersections <= 1 && (numCorners === 3 || (numCorners <= 4 && boxFill >= 0.28 && boxFill <= 0.64))) {
    return { name: "triangle", symbol: "△", type: "shape", confidence: 0.95 };
  }

  // 4. SQUARE / RECTANGLE (Zero intersections, 4 corners, high box fill)
  if (isClosed && intersections <= 1 && boxFill > 0.64) {
    if (Math.abs(width / height - 1.0) < 0.28) {
      return { name: "square", symbol: "□", type: "shape", confidence: 0.95 };
    }
    return { name: "rectangle", symbol: "▭", type: "shape", confidence: 0.93 };
  }

  // 5. DIAMOND
  if (isClosed && intersections <= 1 && boxFill >= 0.38 && boxFill <= 0.65) {
    return { name: "diamond", symbol: "◇", type: "shape", confidence: 0.91 };
  }

  // 6. CARET (^) EXONENT / POWER: Inverted V
  let minYIdx = 0;
  for (let i = 1; i < resampled.length; i++) {
    if (resampled[i].y < resampled[minYIdx].y) minYIdx = i;
  }
  const apexPt = resampled[minYIdx];
  if (!isClosed && minYIdx > 10 && minYIdx < 54 && apexPt.y < startPt.y - height * 0.35 && apexPt.y < endPt.y - height * 0.35) {
    return { name: "caret", symbol: "^", type: "operator", confidence: 0.92 };
  }

  // 7. STRAIGHT LINE / SEGMENT
  if (!isClosed && (totalLength / diagonal) < 1.25) {
    return { name: "line", symbol: "―", type: "shape", confidence: 0.95 };
  }

  // 8. ARROW
  if (!isClosed && totalLength > diagonal * 1.25) {
    return { name: "arrow", symbol: "➔", type: "shape", confidence: 0.91 };
  }

  if (isClosed) return { name: "circle", symbol: "○", type: "shape", confidence: 0.88 };
  return { name: "line", symbol: "―", type: "shape", confidence: 0.88 };
}

// Check Star Geometry
function checkStarGeometry(pts, cx, cy, width, height) {
  const intersections = countStrokeSelfIntersections(pts);
  if (intersections >= 3) return { isStar: true };
  const radii = pts.map(p => Math.hypot(p.x - cx, p.y - cy));
  const meanR = radii.reduce((a, b) => a + b, 0) / pts.length;
  let peaks = 0;
  for (let i = 2; i < pts.length - 2; i++) {
    if (radii[i] > radii[i - 1] && radii[i] > radii[i + 1] && radii[i] > meanR * 1.10) {
      peaks++;
    }
  }
  return { isStar: (peaks >= 3 && Math.abs(width / height - 1.0) < 0.5) || intersections >= 2 };
}

// Update Dashboard UI with latest equation & symbol results
function updateDashboardResults(latestToken) {
  // Build equation expression string
  const eqStr = state.equationTokens.map(t => t.text).join(" ");
  if (resEquation) resEquation.textContent = eqStr || "--";

  if (resShape) {
    if (latestToken.type === "shape" || latestToken.name.toLowerCase().includes("star") || latestToken.name.toLowerCase().includes("circle") || latestToken.name.toLowerCase().includes("triangle") || latestToken.name.toLowerCase().includes("square") || latestToken.name.toLowerCase().includes("diamond") || latestToken.name.toLowerCase().includes("arrow")) {
      resShape.textContent = latestToken.name.toUpperCase();
      if (resShapeConf) resShapeConf.textContent = `Confidence: ${(latestToken.confidence * 100).toFixed(1)}%`;
    } else {
      resShape.textContent = "--";
      if (resShapeConf) resShapeConf.textContent = "Confidence: --";
    }
  }

  if (resChar) {
    if (latestToken.type === "operator" || latestToken.type === "number" || latestToken.type === "variable") {
      resChar.textContent = `${latestToken.text} (${latestToken.name})`;
      if (resCharSub) resCharSub.textContent = latestToken.disambiguation || "Orientation: Canonical";
    } else {
      resChar.textContent = latestToken.symbol || latestToken.text || "--";
      if (resCharSub) resCharSub.textContent = "Shape / Symbol";
    }
  }

  if (resDisambiguation) {
    if (latestToken.disambiguation) {
      resDisambiguation.textContent = latestToken.disambiguation;
    } else {
      resDisambiguation.textContent = `Math Context: Active (${state.equationTokens.length} tokens)`;
    }
  }

  if (resCardsCount) {
    resCardsCount.textContent = `${state.savedCards.length} Cards`;
  }
}

// Autocorrect Symbol on Canvas
function applyAutocorrectToLastSymbol(result, symbolStrokes) {
  const bbox = result.bbox;
  const penColor = document.getElementById("pen-color").value;
  const penSize = parseInt(document.getElementById("pen-size").value, 10);

  const cleanPts = [];
  const { cx, cy, width, height, minX, maxX, minY, maxY } = bbox;

  if (result.text === "○" || result.name.toUpperCase().includes("CIRCLE")) {
    const rx = width / 2;
    const ry = height / 2;
    for (let i = 0; i <= 60; i++) {
      const theta = (i / 60) * Math.PI * 2;
      cleanPts.push({ x: cx + rx * Math.cos(theta), y: cy + ry * Math.sin(theta) });
    }
  } else if (result.text === "+" || result.name.includes("Plus")) {
    // Clean Plus sign: horizontal crossbar + vertical bar
    const arm = Math.min(width, height) * 0.45;
    cleanPts.push({ x: cx - arm, y: cy });
    cleanPts.push({ x: cx + arm, y: cy });
    allStrokes.push([{ x: cx, y: cy - arm }, { x: cx, y: cy + arm }]);
  } else if (result.text === "×" || result.name.includes("Multiplication")) {
    // Clean Multiplication cross
    const arm = Math.min(width, height) * 0.40;
    cleanPts.push({ x: cx - arm, y: cy - arm });
    cleanPts.push({ x: cx + arm, y: cy + arm });
    allStrokes.push([{ x: cx - arm, y: cy + arm }, { x: cx + arm, y: cy - arm }]);
  } else if (result.text === "=" || result.name.includes("Equals")) {
    // Two clean parallel horizontal bars
    const wHalf = width * 0.45;
    const gap = Math.max(6, height * 0.20);
    cleanPts.push({ x: cx - wHalf, y: cy - gap });
    cleanPts.push({ x: cx + wHalf, y: cy - gap });
    allStrokes.push([{ x: cx - wHalf, y: cy + gap }, { x: cx + wHalf, y: cy + gap }]);
  } else if (result.text === "⭐" || result.name.toUpperCase().includes("STAR")) {
    const outerR = Math.max(width, height) / 2;
    const innerR = outerR * 0.40;
    for (let i = 0; i <= 10; i++) {
      const angle = (i * Math.PI) / 5 - Math.PI / 2;
      const r = i % 2 === 0 ? outerR : innerR;
      cleanPts.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
    }
  } else if (result.text === "△" || result.name.toUpperCase().includes("TRIANGLE")) {
    cleanPts.push({ x: cx, y: minY });
    cleanPts.push({ x: maxX, y: maxY });
    cleanPts.push({ x: minX, y: maxY });
    cleanPts.push({ x: cx, y: minY });
  } else if (result.text === "□" || result.name.toUpperCase().includes("SQUARE")) {
    const side = Math.max(width, height);
    const half = side / 2;
    cleanPts.push({ x: cx - half, y: cy - half });
    cleanPts.push({ x: cx + half, y: cy - half });
    cleanPts.push({ x: cx + half, y: cy + half });
    cleanPts.push({ x: cx - half, y: cy + half });
    cleanPts.push({ x: cx - half, y: cy - half });
  } else if (result.text === "▭" || result.name.toUpperCase().includes("RECTANGLE")) {
    cleanPts.push({ x: minX, y: minY });
    cleanPts.push({ x: maxX, y: minY });
    cleanPts.push({ x: maxX, y: maxY });
    cleanPts.push({ x: minX, y: maxY });
    cleanPts.push({ x: minX, y: minY });
  } else if (result.text === "◇" || result.name.toUpperCase().includes("DIAMOND")) {
    cleanPts.push({ x: cx, y: minY });
    cleanPts.push({ x: maxX, y: cy });
    cleanPts.push({ x: cx, y: maxY });
    cleanPts.push({ x: minX, y: cy });
    cleanPts.push({ x: cx, y: minY });
  } else if (result.text === "―" || result.text === "-") {
    cleanPts.push({ x: minX, y: cy });
    cleanPts.push({ x: maxX, y: cy });
  }

  if (cleanPts.length > 0) {
    cleanPts.color = penColor;
    cleanPts.width = penSize;
    // Replace the strokes with the clean vector stroke
    const numStrokesToRemove = symbolStrokes.length;
    allStrokes.splice(allStrokes.length - numStrokesToRemove, numStrokesToRemove, cleanPts);
    redrawAllStrokes();
  }
}

// 📦 FINISH EQUATION: Shrinks and fits neatly into the corner, creating draggable card
function finishCurrentEquationAndSaveCard() {
  if (allStrokes.length === 0 && state.equationTokens.length === 0) return;

  saveCanvasHistory();

  const cardIndex = state.savedCards.length + 1;
  const eqText = state.equationTokens.map(t => t.text).join(" ") || "Equation #" + cardIndex;

  // Create thumbnail of the drawn equation
  const previewCanvas = document.createElement("canvas");
  previewCanvas.width = 240;
  previewCanvas.height = 70;
  const pCtx = previewCanvas.getContext("2d");
  pCtx.fillStyle = "#161f2e";
  pCtx.fillRect(0, 0, 240, 70);

  // Find bounding box of all strokes
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const s of allStrokes) {
    for (const p of s) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  }

  const sW = Math.max(maxX - minX, 1);
  const sH = Math.max(maxY - minY, 1);
  const scale = Math.min(200 / sW, 55 / sH, 1.0);
  const offsetX = (240 - sW * scale) / 2 - minX * scale;
  const offsetY = (70 - sH * scale) / 2 - minY * scale;

  // Draw strokes scaled to thumbnail
  for (const s of allStrokes) {
    if (s.length < 2) continue;
    pCtx.beginPath();
    pCtx.moveTo(s[0].x * scale + offsetX, s[0].y * scale + offsetY);
    for (let i = 1; i < s.length; i++) {
      pCtx.lineTo(s[i].x * scale + offsetX, s[i].y * scale + offsetY);
    }
    pCtx.strokeStyle = s.color || "#4f8cff";
    pCtx.lineWidth = Math.max(2, (s.width || 6) * scale);
    pCtx.lineCap = "round";
    pCtx.stroke();
  }

  // Corner Layout positioning (matching user screenshot indent & header space)
  const initialLeft = 24;
  const initialTop = 24 + (cardIndex - 1) * 125;

  // Create Card DOM Element
  const cardEl = document.createElement("div");
  cardEl.className = "equation-card";
  cardEl.id = `card-${cardIndex}`;
  cardEl.style.left = `${initialLeft}px`;
  cardEl.style.top = `${initialTop}px`;

  cardEl.innerHTML = `
    <div class="card-header">
      <span class="card-title">Equation #${cardIndex}</span>
      <button class="card-close-btn" title="Remove Card">&times;</button>
    </div>
    <div class="card-body">
      <div class="card-text">${eqText}</div>
      <img class="card-canvas-preview" src="${previewCanvas.toDataURL()}" alt="Equation Preview">
      <div class="card-footer">
        <span>✊ Fist to Grab & Drag</span>
        <span>Saved</span>
      </div>
    </div>
  `;

  // Attach mouse drag & drop listeners for desktop
  setupCardMouseDrag(cardEl);

  // Close button
  cardEl.querySelector(".card-close-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    cardEl.remove();
    state.savedCards = state.savedCards.filter(c => c.element !== cardEl);
    resCardsCount.textContent = `${state.savedCards.length} Cards`;
  });

  cardsLayer.appendChild(cardEl);

  const cardObj = {
    id: cardIndex,
    element: cardEl,
    x: initialLeft,
    y: initialTop,
    text: eqText,
  };
  state.savedCards.push(cardObj);

  // Clear main whiteboard canvas for next equation
  allStrokes = [];
  activeStroke = [];
  state.currentSymbolStrokes = [];
  state.equationTokens = [];
  canvasCtx.clearRect(0, 0, whiteboardCanvas.width, whiteboardCanvas.height);

  resCardsCount.textContent = `${state.savedCards.length} Cards`;
  resEquation.textContent = "--";
  resShape.textContent = "Equation Saved to Corner";
  resShapeConf.textContent = `Card #${cardIndex} ready to drag`;
}

// Setup Card Mouse Drag & Drop
function setupCardMouseDrag(cardEl) {
  let isDragging = false;
  let startX, startY, initialLeft, initialTop;

  cardEl.addEventListener("mousedown", (e) => {
    if (e.target.classList.contains("card-close-btn")) return;
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    initialLeft = parseInt(cardEl.style.left, 10) || 0;
    initialTop = parseInt(cardEl.style.top, 10) || 0;
    cardEl.classList.add("grabbed");
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    cardEl.style.left = `${Math.max(0, initialLeft + dx)}px`;
    cardEl.style.top = `${Math.max(0, initialTop + dy)}px`;
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      cardEl.classList.remove("grabbed");
    }
  });
}

// Mouse / Touch drawing on whiteboard canvas
let isMouseDown = false;
function setupCanvasDrawing() {
  function resizeCanvas() {
    const rect = whiteboardCanvas.parentElement.getBoundingClientRect();
    whiteboardCanvas.width = rect.width;
    whiteboardCanvas.height = rect.height;
    redrawAllStrokes();
  }
  window.addEventListener("resize", resizeCanvas);
  setTimeout(resizeCanvas, 50);

  whiteboardCanvas.addEventListener("mousedown", (e) => {
    isMouseDown = true;
    const rect = whiteboardCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    activeStroke = [{ x, y }];

    if (state.symbolDebounceTimer) {
      clearTimeout(state.symbolDebounceTimer);
      state.symbolDebounceTimer = null;
    }
  });

  whiteboardCanvas.addEventListener("mousemove", (e) => {
    if (!isMouseDown) return;
    const rect = whiteboardCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const prev = activeStroke[activeStroke.length - 1];
    canvasCtx.beginPath();
    canvasCtx.moveTo(prev.x, prev.y);
    canvasCtx.lineTo(x, y);
    canvasCtx.strokeStyle = document.getElementById("pen-color").value;
    canvasCtx.lineWidth = parseInt(document.getElementById("pen-size").value, 10);
    canvasCtx.lineCap = "round";
    canvasCtx.lineJoin = "round";
    canvasCtx.stroke();

    activeStroke.push({ x, y });
  });

  window.addEventListener("mouseup", () => {
    if (isMouseDown) {
      isMouseDown = false;
      if (activeStroke.length >= 4) {
        saveCanvasHistory();
        allStrokes.push([...activeStroke]);
        state.currentSymbolStrokes.push([...activeStroke]);
        redrawAllStrokes();

        if (state.symbolDebounceTimer) clearTimeout(state.symbolDebounceTimer);
        state.symbolDebounceTimer = setTimeout(() => {
          finalizeCurrentSymbol();
        }, 1000);
      }
      activeStroke = [];
    }
  });

  document.getElementById("btn-clear").addEventListener("click", () => {
    saveCanvasHistory();
    allStrokes = [];
    activeStroke = [];
    state.currentSymbolStrokes = [];
    state.equationTokens = [];
    canvasCtx.clearRect(0, 0, whiteboardCanvas.width, whiteboardCanvas.height);
    resEquation.textContent = "--";
    resShape.textContent = "Draw something...";
    resShapeConf.textContent = "Confidence: --";
  });

  btnFinishEq.addEventListener("click", () => {
    finishCurrentEquationAndSaveCard();
  });

  btnCancelFinish.addEventListener("click", () => {
    state.handOffStartTime = null;
    state.autoFinishCancelled = true;
    autofinishToast.classList.remove("visible");
  });

  document.getElementById("btn-recognize").addEventListener("click", () => {
    finalizeCurrentSymbol();
  });

  // Autocorrect Toggle
  btnAutocorrect.addEventListener("click", () => {
    state.autocorrect = !state.autocorrect;
    btnAutocorrect.textContent = state.autocorrect ? "✨ Autocorrect: ON" : "✨ Autocorrect: OFF";
    btnAutocorrect.classList.toggle("active", state.autocorrect);
  });

  // Undo button
  btnUndo.addEventListener("click", undoLastStroke);
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "z") {
      e.preventDefault();
      undoLastStroke();
    }
  });
}

function saveCanvasHistory() {
  state.strokeHistory.push(JSON.parse(JSON.stringify(allStrokes)));
  if (state.strokeHistory.length > 20) state.strokeHistory.shift();
}

function undoLastStroke() {
  if (state.strokeHistory.length > 0) {
    allStrokes = state.strokeHistory.pop();
    redrawAllStrokes();
    if (state.equationTokens.length > 0) {
      state.equationTokens.pop();
      const eqStr = state.equationTokens.map(t => t.text).join(" ");
      resEquation.textContent = eqStr || "--";
    }
  }
}

function redrawAllStrokes() {
  canvasCtx.clearRect(0, 0, whiteboardCanvas.width, whiteboardCanvas.height);
  const penColor = document.getElementById("pen-color").value;
  const penSize = parseInt(document.getElementById("pen-size").value, 10);

  for (const s of allStrokes) {
    if (s.length < 2) continue;
    canvasCtx.beginPath();
    canvasCtx.moveTo(s[0].x, s[0].y);
    for (let i = 1; i < s.length; i++) {
      canvasCtx.lineTo(s[i].x, s[i].y);
    }
    canvasCtx.strokeStyle = s.color || penColor;
    canvasCtx.lineWidth = s.width || penSize;
    canvasCtx.lineCap = "round";
    canvasCtx.lineJoin = "round";
    canvasCtx.stroke();
  }
}

// Resample polyline to exactly n points using cumulative arc-length parameterization
function resampleStroke(pts, n = 64) {
  if (pts.length === 0) return Array(n).fill({ x: 0, y: 0 });
  if (pts.length === 1) return Array(n).fill(pts[0]);

  let totalDist = 0;
  const dists = [0];
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    totalDist += d;
    dists.push(totalDist);
  }

  if (totalDist < 1e-4) return Array(n).fill(pts[0]);

  const step = totalDist / (n - 1);
  const resampled = [];
  let curIdx = 0;

  for (let i = 0; i < n; i++) {
    const targetDist = i * step;
    while (curIdx < dists.length - 1 && dists[curIdx + 1] < targetDist) {
      curIdx++;
    }
    if (curIdx >= dists.length - 1) {
      resampled.push(pts[pts.length - 1]);
    } else {
      const segLen = dists[curIdx + 1] - dists[curIdx];
      const t = segLen > 0 ? (targetDist - dists[curIdx]) / segLen : 0;
      resampled.push({
        x: pts[curIdx].x + t * (pts[curIdx + 1].x - pts[curIdx].x),
        y: pts[curIdx].y + t * (pts[curIdx + 1].y - pts[curIdx].y),
      });
    }
  }
  return resampled;
}

// In-App Recorder
function setupRecorder() {
  const btnCapture = document.getElementById("btn-rec-capture");
  const btnExport = document.getElementById("btn-rec-export");
  const recTableBody = document.getElementById("rec-table-body");

  btnCapture.addEventListener("click", captureSample);
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      captureSample();
    }
  });

  function captureSample() {
    if (!state.lastLandmarks) {
      alert("No active hand detected to record. Turn on camera first.");
      return;
    }
    const label = document.getElementById("rec-label").value.trim();
    const signer = document.getElementById("rec-signer").value.trim();
    const category = document.getElementById("rec-category").value;

    const sample = {
      label,
      signer,
      category,
      timestamp: Date.now(),
      landmarks: state.lastLandmarks.map((p) => [p.x, p.y, p.z]),
    };
    state.recordedSamples.push(sample);
    updateRecorderTable();
  }

  function updateRecorderTable() {
    const counts = {};
    for (const s of state.recordedSamples) {
      counts[s.label] = (counts[s.label] || 0) + 1;
    }
    recTableBody.innerHTML = Object.entries(counts)
      .map(
        ([lbl, count]) => `
      <tr>
        <td><strong>${lbl}</strong></td>
        <td>${count} samples</td>
        <td>${document.getElementById("rec-signer").value}</td>
        <td><span class="badge badge-info">Captured</span></td>
      </tr>
    `
      )
      .join("");
  }

  btnExport.addEventListener("click", () => {
    if (state.recordedSamples.length === 0) {
      alert("No recorded samples to export.");
      return;
    }
    const blob = new Blob([JSON.stringify(state.recordedSamples, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `camboard_recorded_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

// In-Browser Benchmark Runner
async function runBrowserBenchmarks() {
  const container = document.getElementById("benchmarks-container");
  container.innerHTML = "<p>Running in-browser WebAssembly benchmark (50 warmup + 200 runs)...</p>";

  const benchmarkList = [
    { name: "Gesture Classifier (MLP)", path: "models/gesture.onnx", shape: [1, 63] },
    { name: "ASL Fingerspelling (MLP)", path: "models/fingerspelling.onnx", shape: [1, 63] },
    { name: "Motion Letters J/Z (GRU)", path: "models/motion_letters.onnx", shape: [1, 30, 126] },
    { name: "Stroke Shapes (1D-CNN)", path: "models/stroke_shapes.onnx", shape: [1, 3, 64] },
    { name: "Stroke Characters (BiGRU)", path: "models/stroke_chars.onnx", shape: [1, 64, 3] },
    { name: "Sign Vocabulary (Transformer)", path: "models/sign_vocab.onnx", shape: [1, 64, 132] },
  ];

  const results = [];

  for (const b of benchmarkList) {
    try {
      const sess = await ort.InferenceSession.create(b.path);
      const totalElements = b.shape.reduce((a, c) => a * c, 1);
      const dummy = new Float32Array(totalElements);
      for (let i = 0; i < totalElements; i++) dummy[i] = Math.random();
      const tensor = new ort.Tensor("float32", dummy, b.shape);
      const feeds = { [sess.inputNames[0]]: tensor };

      // Warmup 50 runs
      for (let w = 0; w < 50; w++) await sess.run(feeds);

      // Timed 200 runs
      const timings = [];
      for (let i = 0; i < 200; i++) {
        const t0 = performance.now();
        await sess.run(feeds);
        timings.push(performance.now() - t0);
      }
      timings.sort((a, b) => a - b);
      const p50 = timings[Math.floor(timings.length * 0.5)].toFixed(2);
      const p95 = timings[Math.floor(timings.length * 0.95)].toFixed(2);

      results.push({ name: b.name, p50, p95, status: "PASS" });
    } catch (e) {
      results.push({ name: b.name, p50: "N/A", p95: "N/A", status: "Error: " + e.message });
    }
  }

  container.innerHTML = results
    .map(
      (r) => `
    <div class="bench-card">
      <h4>${r.name}</h4>
      <div class="bench-stat"><span>P50 Latency:</span> <strong>${r.p50} ms</strong></div>
      <div class="bench-stat"><span>P95 Latency:</span> <strong>${r.p95} ms</strong></div>
      <div class="bench-stat"><span>Runtime Backend:</span> <code>WebAssembly / CPU</code></div>
    </div>
  `
    )
    .join("");
}

// Navigation Tabs
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const targetId = tab.dataset.tab;
      document.getElementById(targetId).classList.add("active");
    });
  });

  const matrix = document.getElementById("matrix-gestures");
  matrix.innerHTML = GESTURE_CLASSES.map(
    (g) => `
    <div class="matrix-item" data-gesture="${g}">
      <strong>${g}</strong>
    </div>
  `
  ).join("");

  const alphabetGrid = document.getElementById("grid-asl-letters");
  alphabetGrid.innerHTML = ASL_CLASSES.map(
    (l) => `
    <div class="letter-badge" data-letter="${l}">${l}</div>
  `
  ).join("");
}

// Entrypoint
window.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupCanvasDrawing();
  setupRecorder();

  document.getElementById("btn-toggle-cam").addEventListener("click", toggleCamera);
  document.getElementById("btn-run-benchmarks").addEventListener("click", runBrowserBenchmarks);

  initMediaPipe();
  await initONNXModels();
});
