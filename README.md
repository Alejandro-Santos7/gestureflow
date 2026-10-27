# GestureFlow Suite – Fase 1

Control del cursor de Windows con la mano. Todo local: ningún fotograma se guarda ni sale del equipo
(los datos de entrenamiento son solo coordenadas de landmarks).

## Instalar (Windows 10/11, Python 3.13, una webcam)
```
git clone https://github.com/Alejandro-Santos7/gestureflow.git
cd gestureflow
setup.bat
```
`setup.bat` crea el entorno `.venv`, instala las dependencias (versiones fijadas en `requirements.txt`) y crea el
acceso directo **GestureFlow.lnk** con las rutas de tu equipo. Se puede repetir sin problema.
`run.bat` también sirve: si falta el entorno, ejecuta `setup.bat` antes de abrir.

## Primer arranque
1. Doble clic en **GestureFlow.lnk** (sin consola).
2. La app ofrece descargar el modelo de manos de MediaPipe (`models/hand_landmarker.task`, ~7.8 MB, una sola vez).
3. Se abre el **Tutorial**. El último paso graba **3 rondas guiadas** (≈ 3 min) y, al terminar, **entrena tu modelo
   personal automáticamente** con tus datos. Hasta entonces funciona con reglas geométricas.
4. Cada grabación nueva en Entrenamiento vuelve a entrenar con todos tus datos (desactivable: «Entrenar al terminar
   cada grabación»).

## Qué queda solo en tu equipo
No se sube al repositorio (`.gitignore`) y se crea al usar la app:

| Ruta | Contenido |
|---|---|
| `config.json` | tus ajustes (se crea al guardar; si falta, valores por defecto) |
| `data/raw/`, `data/discarded/` | tus grabaciones: coordenadas de la mano, nunca imágenes |
| `runs/` | registro de cada entrenamiento (métricas, huella de los datos) |
| `models/` | modelo de MediaPipe descargado y tu modelo personal `latest.npz` |
| `.venv/`, `GestureFlow.lnk` | entorno y acceso directo con rutas de tu equipo (`setup.bat`) |

## Interfaz (diseño en `design/desktop.dc.html`)
| Vista | Qué hace |
|---|---|
| Monitor | Cámara con esqueleto de la mano, zona activa, telemetría real (FPS, latencia, estabilidad, luz, CPU/RAM), mano preferida, recentrar |
| Entrenamiento | Grabación guiada, gestión de datos (descartar = mover a `data/discarded`), entrenar, veredicto e historial |
| Configuración | Sensibilidad, velocidad máxima, zona muerta, filtro One Euro, modos de clic, dwell, antirrebote, sonido/halo |
| Mapeo | Acción por gesto (izq./der./doble/arrastrar), probar gestos en vivo, confianza mínima del pellizco |
| Laboratorio | Luz de la imagen + autocalibración de exposición, dianas de puntería, guía de posturas |
| Hoja de ruta / Tutorial | Fase 1 vs Fase 2; tutorial interactivo de 6 pasos |

Paneles flotantes: **HUD** (cámara mini, próximo clic, sensibilidad, pausa), **Barra de acciones**
(arma clic derecho/doble/central/arrastrar para el próximo clic, teclado en pantalla de Windows, bloqueo) y
**menú de bandeja** (clic en el icono). Un anillo junto al cursor muestra el progreso del dwell.

Si hay dos «manos» en cámara (MediaPipe a veces toma la cara por una), se sigue la más cercana a la que ya se seguía; una «mano» que aparece de golpe lejos tiene que mantenerse 4 fotogramas para ser aceptada.

Parada de emergencia global: **F9** o **Ctrl+Alt+P**.

**Cerrar y cámara:** la **X** cierra GestureFlow por completo (1 clic, ~0.5 s) y apaga la cámara; minimizar (—) la deja en
la barra de tareas. Solo se permite una instancia: abrirla otra vez trae la ventana existente al frente.
Si la cámara falla o deja de enviar datos, la app la reinicia sola (hasta 3 intentos) y muestra un aviso con el botón
**Reiniciar cámara**, que también está en Monitor y en el menú de la bandeja.

## Fluidez del cursor
- **Cámara a 30 FPS aun con poca luz:** con la exposición automática las webcams bajan a 4–15 FPS en una habitación
  oscura. GestureFlow usa exposición manual corta (≈16 ms, menos desenfoque al mover la mano) + ganancia electrónica al máximo, y la acorta sola si la
  cámara no llega a 26 FPS. Mide en Monitor → Tasa de refresco.
- **Cursor a ~120 Hz:** un hilo aparte mueve el ratón cada 8 ms hacia la posición detectada (con interpolación y una
  predicción corta), así que se ve fluido aunque la detección llegue a 30 Hz. El clic se hace exactamente donde está el dedo.
- **Clic sin desvío:** el cursor sigue el centro de los nudillos, no la punta del índice. Medido con las grabaciones de
  `data/raw`: al pellizcar, la punta del índice se desplaza ~330 px de pantalla y los nudillos ~57 px, con la mitad de
  temblor. Además el clic vuelve al punto en que los dedos empezaron a cerrarse, necesita 2 fotogramas seguidos de
  pellizco (uno suelto no hace clic) y no se repite hasta reabrir la pinza. Configuración → «Punto que mueve el cursor»
  permite volver a la punta del índice.
- Ajustes en Configuración: «Intensidad del filtro» (más bajo = más suave) y «Reactividad al mover rápido» (más alto = menos retardo).
- `config.json`: `"fast_camera": false` desactiva el ajuste de cámara; `"exposure"` cambia la exposición inicial.

## Gestos
| Gesto | Acción (configurable en Mapeo) |
|---|---|
| puntero (índice) | la mano mueve el cursor (lo guían los nudillos, no la punta del dedo); quieto = clic por permanencia |
| pinch | clic en el punto donde estaba el cursor antes de cerrar los dedos; el cursor espera hasta que se reabren |
| palma abierta y quieta ~0.8 s | pausa de descanso; moverla la quita (una mano relajada se ve como palma y no debe congelar el cursor) |
| reposo (puño) | sin acción |

## Entrenamiento por consola (opcional)
`python -m gestureflow.collect --user ana` y `python -m gestureflow.train`.
Registros en `runs/<fecha>_<nombre>/` (config, métricas por sesión, SHA-256 de los datos, entorno) y `runs/index.csv`.

## Pruebas
`.venv\Scripts\python tests\test_core.py`

## Limitaciones conocidas
- No se puede inyectar ratón sobre ventanas ejecutadas como administrador (UIPI) salvo que GestureFlow también lo sea.
- La luminosidad es la de la imagen (0–100 %), no lux reales: la webcam no mide lux.
- Un modelo entrenado con una sola persona generaliza peor a otras.
