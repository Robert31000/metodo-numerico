import React, { useState, useRef, useCallback } from 'react';

interface Tensor {
  H: number;
  W: number;
  C: number;
  data: Float32Array;
}

interface InpaintResult {
  reconstructed: Tensor;
  residuals: number[];
  rmse: number;
  iterations: number;
}

// ============================================================================
// UTILIDADES DE IMAGEN
// ============================================================================

const rgbaToTensor = (imageData: ImageData): Tensor => {
  const { width: W, height: H, data: rgba } = imageData;
  const C = 3;
  const tensor: Tensor = {
    H, W, C,
    data: new Float32Array(H * W * C)
  };
  
  for (let i = 0; i < H * W; i++) {
    tensor.data[i * 3 + 0] = rgba[i * 4 + 0] / 255;
    tensor.data[i * 3 + 1] = rgba[i * 4 + 1] / 255;
    tensor.data[i * 3 + 2] = rgba[i * 4 + 2] / 255;
  }
  
  return tensor;
};

const tensorToImageData = (tensor: Tensor): ImageData => {
  const { H, W, data } = tensor;
  const imageData = new ImageData(W, H);
  
  for (let i = 0; i < H * W; i++) {
    imageData.data[i * 4 + 0] = Math.round(Math.min(255, Math.max(0, data[i * 3 + 0] * 255)));
    imageData.data[i * 4 + 1] = Math.round(Math.min(255, Math.max(0, data[i * 3 + 1] * 255)));
    imageData.data[i * 4 + 2] = Math.round(Math.min(255, Math.max(0, data[i * 3 + 2] * 255)));
    imageData.data[i * 4 + 3] = 255;
  }
  
  return imageData;
};

const makeDamagedView = (original: Tensor, known: Uint8Array): Tensor => {
  const damaged: Tensor = {
    H: original.H,
    W: original.W,
    C: original.C,
    data: new Float32Array(original.data)
  };
  
  // Píxeles con known=0 → negro
  for (let i = 0; i < original.H * original.W; i++) {
    if (known[i] === 0) {
      damaged.data[i * 3 + 0] = 0;
      damaged.data[i * 3 + 1] = 0;
      damaged.data[i * 3 + 2] = 0;
    }
  }
  
  return damaged;
};

const randomKnownMask = (W: number, H: number, damagePercent: number): Uint8Array => {
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    mask[i] = Math.random() > damagePercent / 100 ? 1 : 0;
  }
  return mask;
};

const inpaintGaussSeidel = (
  damaged: Tensor,
  known: Uint8Array,
  lambda: number,
  beta: number,
  maxIter: number,
  tol: number
): InpaintResult => {
  const { H, W, C } = damaged;
  const U = new Float32Array(damaged.data);
  const residuals: number[] = [];
  const omega = 1.25; // Factor SOR para acelerar convergencia
  
  let iter = 0;
  
  for (iter = 0; iter < maxIter; iter++) {
    let totalChange = 0;
    let changeCount = 0;
    
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;
        
        if (known[idx] === 1) {
          // Píxel conocido: clamp
          for (let c = 0; c < C; c++) {
            U[idx * C + c] = damaged.data[idx * C + c];
          }
        } else {
          // Píxel desconocido: promedio 4-vecinos con espejo
          const neighbors = [
            [Math.max(0, y - 1), x],           // Arriba
            [Math.min(H - 1, y + 1), x],       // Abajo
            [y, Math.max(0, x - 1)],           // Izquierda
            [y, Math.min(W - 1, x + 1)]        // Derecha
          ];
          
          for (let c = 0; c < C; c++) {
            let sum = 0;
            for (const [ny, nx] of neighbors) {
              const nidx = ny * W + nx;
              sum += U[nidx * C + c];
            }
            
            const oldVal = U[idx * C + c];
            const newVal = sum / 4;
            
            // SOR: acelera convergencia
            U[idx * C + c] = oldVal + omega * (newVal - oldVal);
            
            totalChange += Math.abs(U[idx * C + c] - oldVal);
            changeCount++;
          }
        }
      }
    }
    
    const avgChange = changeCount > 0 ? totalChange / changeCount : 0;
    residuals.push(avgChange);
    
    if (avgChange < tol) {
      break;
    }
  }
  
  const reconstructed: Tensor = { H, W, C, data: U };
  const rmse = rmseOnMissing(damaged, reconstructed, known);
  
  return { reconstructed, residuals, rmse, iterations: iter + 1 };
};

const rmseOnMissing = (original: Tensor, reconstructed: Tensor, known: Uint8Array): number => {
  const { H, W, C } = original;
  let sumSqError = 0;
  let count = 0;
  
  for (let i = 0; i < H * W; i++) {
    if (known[i] === 0) {
      for (let c = 0; c < C; c++) {
        const diff = original.data[i * C + c] - reconstructed.data[i * C + c];
        sumSqError += diff * diff;
      }
      count += C;
    }
  }
  
  return count > 0 ? Math.sqrt(sumSqError / count) : 0;
};

interface MaskCanvasProps {
  width: number;
  height: number;
  imageData: ImageData | null;
  onMaskChange: (mask: Uint8Array) => void;
  brushSize: number;
}

const MaskCanvas: React.FC<MaskCanvasProps> = ({ width, height, imageData, onMaskChange, brushSize }) => {
  const baseRef = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const maskRef = useRef<Uint8Array>(new Uint8Array(width * height));

  // Inicializar máscara y canvas base cuando cambia la imagen
  React.useEffect(() => {
    maskRef.current = new Uint8Array(width * height); // Todos en 0 (desconocidos al pintar)
    
    const canvas = baseRef.current;
    if (!canvas || !imageData) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.putImageData(imageData, 0, 0);
    
    // Limpiar overlay
    const drawCanvas = drawRef.current;
    if (drawCanvas) {
      const drawCtx = drawCanvas.getContext('2d');
      if (drawCtx) {
        drawCtx.clearRect(0, 0, width, height);
      }
    }
  }, [width, height, imageData]);

  const drawMask = (clientX: number, clientY: number) => {
    const canvas = drawRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const px = Math.floor((clientX - rect.left) * scaleX);
    const py = Math.floor((clientY - rect.top) * scaleY);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const radius = Math.floor(brushSize / 2);
    
    // Pintar en máscara (1 = pintado/desconocido)
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy <= radius * radius) {
          const nx = px + dx;
          const ny = py + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            maskRef.current[ny * width + nx] = 1;
          }
        }
      }
    }

    // Dibujar overlay rojo
    ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.fillRect(px - radius, py - radius, brushSize, brushSize);
    
    onMaskChange(maskRef.current);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDrawing(true);
    drawMask(e.clientX, e.clientY);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    e.preventDefault();
    if (isDrawing) {
      drawMask(e.clientX, e.clientY);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handleClearMask = () => {
    maskRef.current.fill(0);
    const canvas = drawRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, width, height);
      }
    }
    onMaskChange(maskRef.current);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ position: 'relative', width: 'fit-content' }}>
        {/* Canvas base (imagen) */}
        <canvas
          ref={baseRef}
          width={width}
          height={height}
          style={{
            display: 'block',
            width: '100%',
            maxWidth: '500px',
            border: '2px solid #333',
            zIndex: 1,
            pointerEvents: 'none'
          }}
        />
        {/* Canvas overlay (máscara roja) */}
        <canvas
          ref={drawRef}
          width={width}
          height={height}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            maxWidth: '500px',
            cursor: 'crosshair',
            zIndex: 2,
            pointerEvents: 'auto'
          }}
        />
      </div>
      <button onClick={handleClearMask} style={{ padding: '8px', cursor: 'pointer' }}>
        Limpiar Máscara
      </button>
    </div>
  );
};

interface ResidualChartProps {
  residuals: number[];
  width: number;
  height: number;
}

const ResidualChart: React.FC<ResidualChartProps> = ({ residuals, width, height }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || residuals.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    const maxVal = Math.max(...residuals, 0.001);
    const padding = 40;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;

    // Ejes
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Curva
    ctx.strokeStyle = '#0066cc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    residuals.forEach((val, i) => {
      const x = padding + (i / Math.max(residuals.length - 1, 1)) * graphWidth;
      const y = height - padding - (val / maxVal) * graphHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#000';
    ctx.font = '12px monospace';
    ctx.fillText('Iteraciones', width / 2 - 30, height - 5);
    ctx.save();
    ctx.translate(10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Residuo', 0, 0);
    ctx.restore();
  }, [residuals, width, height]);

  return <canvas ref={canvasRef} width={width} height={height} style={{ border: '1px solid #ccc' }} />;
};

const App: React.FC = () => {
  const [originalImage, setOriginalImage] = useState<HTMLImageElement | null>(null);
  const [originalTensor, setOriginalTensor] = useState<Tensor | null>(null);
  const [manualMask, setManualMask] = useState<Uint8Array | null>(null);
  const [knownMask, setKnownMask] = useState<Uint8Array | null>(null);
  const [damagedTensor, setDamagedTensor] = useState<Tensor | null>(null);
  const [reconstructedTensor, setReconstructedTensor] = useState<Tensor | null>(null);
  const [residuals, setResiduals] = useState<number[]>([]);
  const [rmse, setRmse] = useState<number>(0);
  const [iterations, setIterations] = useState<number>(0);

  // Parámetros
  const [useManualMask, setUseManualMask] = useState(true);
  const [damagePercent, setDamagePercent] = useState(30);
  const [lambda, setLambda] = useState(0.5);
  const [beta, setBeta] = useState(1.5);
  const [maxIter, setMaxIter] = useState(1000);
  const [tol, setTol] = useState(0.0001);
  const [brushSize, setBrushSize] = useState(15);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      const tensor = rgbaToTensor(imageData);

      setOriginalImage(img);
      setOriginalTensor(tensor);
      setManualMask(new Uint8Array(img.width * img.height));
      setKnownMask(null);
      setDamagedTensor(null);
      setReconstructedTensor(null);
      setResiduals([]);
    };

    img.src = URL.createObjectURL(file);
  };

  const handleMaskChange = useCallback((mask: Uint8Array) => {
    setManualMask(mask);
  }, []);

  const handleApplyDamage = () => {
    if (!originalTensor) return;

    let known: Uint8Array;

    if (useManualMask && manualMask) {
      // Contar píxeles pintados
      const paintedCount = manualMask.reduce((sum, val) => sum + val, 0);
      
      if (paintedCount === 0) {
        alert('⚠️ No hay píxeles pintados en la máscara.\n\nPinta las grietas en rojo y vuelve a intentar.');
        return;
      }

      // Invertir: pintado (1) → desconocido (0), no pintado (0) → conocido (1)
      known = new Uint8Array(manualMask.length);
      for (let i = 0; i < manualMask.length; i++) {
        known[i] = manualMask[i] === 1 ? 0 : 1;
      }

      console.log(`✓ Máscara manual aplicada: ${paintedCount} píxeles marcados como desconocidos`);
    } else {
      // Daño aleatorio
      known = randomKnownMask(originalTensor.W, originalTensor.H, damagePercent);
      console.log(`✓ Daño aleatorio: ${damagePercent}%`);
    }

    setKnownMask(known);
    const damaged = makeDamagedView(originalTensor, known);
    setDamagedTensor(damaged);
    setReconstructedTensor(null);
    setResiduals([]);
  };

  const handleReconstruct = () => {
    if (!damagedTensor || !knownMask) return;

    console.log(`🔄 Iniciando reconstrucción: λ=${lambda}, β=${beta}, maxIter=${maxIter}, tol=${tol}`);
    
    const result = inpaintGaussSeidel(damagedTensor, knownMask, lambda, beta, maxIter, tol);
    
    setReconstructedTensor(result.reconstructed);
    setResiduals(result.residuals);
    setRmse(result.rmse);
    setIterations(result.iterations);

    console.log(`✓ Reconstrucción completada: ${result.iterations} iteraciones, RMSE=${result.rmse.toFixed(4)}`);
  };

  const getImageData = (tensor: Tensor | null): ImageData | null => {
    return tensor ? tensorToImageData(tensor) : null;
  };

  const paintedPixels = manualMask ? manualMask.reduce((s, v) => s + v, 0) : 0;
  const unknownPixels = knownMask ? knownMask.filter(v => v === 0).length : 0;

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', maxWidth: '1400px', margin: '0 auto', padding: '20px' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '10px' }}>
         Restauración de Imágenes - Inpainting
      </h1>
      <p style={{ textAlign: 'center', color: '#666', marginBottom: '30px' }}>
        Algoritmo Gauss-Seidel con SOR (ω=1.25)
      </p>

      {/* Controles */}
      <div style={{ marginBottom: '20px', padding: '20px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            📁 Cargar Imagen:
          </label>
          <input type="file" accept="image/png,image/jpeg" onChange={handleImageUpload} />
        </div>

        {originalImage && (
          <>
            {/* Modo de daño */}
            <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#fff', borderRadius: '6px', border: '2px solid #ddd' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontWeight: 'bold' }}>
                <input
                  type="checkbox"
                  checked={useManualMask}
                  onChange={(e) => setUseManualMask(e.target.checked)}
                />
                 Usar máscara manual (pinta grietas en rojo)
              </label>
              
              {useManualMask && paintedPixels > 0 && (
                <p style={{ marginTop: '10px', padding: '8px', backgroundColor: '#e3f2fd', borderRadius: '4px', fontSize: '14px' }}>
                  ✓ <b>{paintedPixels}</b> píxeles pintados
                </p>
              )}

              {!useManualMask && (
                <div style={{ marginTop: '15px' }}>
                  <label style={{ display: 'block', marginBottom: '5px' }}>
                    Daño aleatorio: {damagePercent}%
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    value={damagePercent}
                    onChange={(e) => setDamagePercent(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
            </div>

            {/* Parámetros del algoritmo */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '20px' }}>
              {useManualMask && (
                <div>
                  <label style={{ display: 'block', marginBottom: '5px' }}>🖌️ Tamaño Pincel: {brushSize}px</label>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    value={brushSize}
                    onChange={(e) => setBrushSize(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
              <div>
                <label style={{ display: 'block', marginBottom: '5px' }}>λ (Fidelidad): {lambda.toFixed(2)}</label>
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={lambda}
                  onChange={(e) => setLambda(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '5px' }}>β (Suavidad): {beta.toFixed(2)}</label>
                <input
                  type="range"
                  min="0.5"
                  max="3"
                  step="0.1"
                  value={beta}
                  onChange={(e) => setBeta(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '5px' }}>Iter. Máx: {maxIter}</label>
                <input
                  type="range"
                  min="100"
                  max="2000"
                  step="100"
                  value={maxIter}
                  onChange={(e) => setMaxIter(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '5px' }}>Tolerancia: {tol.toFixed(4)}</label>
                <input
                  type="range"
                  min="0.0001"
                  max="0.5"
                  step="0.0001"
                  value={tol}
                  onChange={(e) => setTol(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* Botones de acción */}
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              <button
                onClick={handleApplyDamage}
                style={{
                  padding: '12px 24px',
                  cursor: 'pointer',
                  backgroundColor: '#ff5722',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontWeight: 'bold',
                  fontSize: '16px'
                }}
              >
                 Aplicar Daño
              </button>
              <button
                onClick={handleReconstruct}
                disabled={!damagedTensor}
                style={{
                  padding: '12px 24px',
                  cursor: damagedTensor ? 'pointer' : 'not-allowed',
                  backgroundColor: damagedTensor ? '#4CAF50' : '#ccc',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontWeight: 'bold',
                  fontSize: '16px'
                }}
              >
                 Reconstruir
              </button>
            </div>

            {unknownPixels > 0 && (
              <p style={{ marginTop: '15px', padding: '10px', backgroundColor: '#fff3cd', borderRadius: '4px' }}>
                ℹ️ Zona a reconstruir: <b>{unknownPixels}</b> píxeles desconocidos
              </p>
            )}
          </>
        )}
      </div>

      {/* Visualización */}
      {originalImage && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '20px' }}>
          <div>
            <h3>📷 Original {useManualMask && '+ Máscara'}</h3>
            {useManualMask ? (
              <MaskCanvas
                width={originalImage.width}
                height={originalImage.height}
                imageData={getImageData(originalTensor)}
                onMaskChange={handleMaskChange}
                brushSize={brushSize}
              />
            ) : (
              <canvas
                ref={(canvas) => {
                  if (canvas && originalTensor) {
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                      const imgData = getImageData(originalTensor);
                      if (imgData) ctx.putImageData(imgData, 0, 0);
                    }
                  }
                }}
                width={originalImage.width}
                height={originalImage.height}
                style={{ border: '2px solid #333', maxWidth: '100%' }}
              />
            )}
          </div>

          {damagedTensor && (
            <div>
              <h3> Vista Dañada</h3>
              <canvas
                ref={(canvas) => {
                  if (canvas) {
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                      const imgData = getImageData(damagedTensor);
                      if (imgData) ctx.putImageData(imgData, 0, 0);
                    }
                  }
                }}
                width={originalImage.width}
                height={originalImage.height}
                style={{ border: '2px solid #333', maxWidth: '100%' }}
              />
            </div>
          )}

          {reconstructedTensor && (
            <div>
              <h3> Reconstruida</h3>
              <canvas
                ref={(canvas) => {
                  if (canvas) {
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                      const imgData = getImageData(reconstructedTensor);
                      if (imgData) ctx.putImageData(imgData, 0, 0);
                    }
                  }
                }}
                width={originalImage.width}
                height={originalImage.height}
                style={{ border: '2px solid #4CAF50', maxWidth: '100%' }}
              />
              <div style={{ marginTop: '10px', padding: '10px', backgroundColor: '#e8f5e9', borderRadius: '4px' }}>
                <p style={{ margin: 0 }}>
                  <strong>⏱Iteraciones:</strong> {iterations} | <strong>RMSE:</strong> {rmse.toFixed(4)}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Gráfico de convergencia */}
      {residuals.length > 0 && (
        <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
          <h3>Curva de Convergencia</h3>
          <ResidualChart residuals={residuals} width={600} height={300} />
        </div>
      )}
    </div>
  );
};

export default App;