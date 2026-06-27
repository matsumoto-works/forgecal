# I Built a Free Browser-Based Forging Simulator in One Month — Here's Why

*When your company's only CAE license is occupied, you build your own tool.*

---

## The Problem: One License, One Seat

Our company uses commercial forging CAE software. It's powerful — and expensive. With only one license shared across the team, there are days when you simply can't run a simulation. You're designing a die, you need a rough load estimate to select the right press, but the seat is taken.

The workaround? Spreadsheets. Hand calculations. Engineering intuition.

That was fine for years. But as designs got more complex, I kept thinking: *there has to be a better way.*

So I built one.

---

## What Is ForgeCal?

**ForgeCal** (https://fem.matsumoto-works.jp) is a free, browser-based 2D finite element solver for cold forging and extrusion simulation. No installation. No license fee. Upload three DXF files, set your material and friction, click Run — and get a forming load estimate in about a minute.

It took one month to build the first working version. Development is ongoing.

---

## Who Is It For?

ForgeCal is designed for three types of users:

- **Die designers** who need a quick load estimate during the early design phase — before committing to a full simulation
- **Students** learning metal forming who want hands-on FEM experience without expensive software
- **Engineers** who need rough-order answers: *Will a 500-ton press be enough? What happens if I change the reduction ratio?*

If you need to answer these questions quickly and cheaply, ForgeCal is for you.

---

## How It Works: Three Files, One Click

The input format is intentionally simple: **three DXF files** — one for the blank (billet), one for the punch, and one for the die.

### Step 1: Draw the geometry in any free CAD tool

No special CAD software is required. I used **Onshape** — a free, browser-based CAD tool — to draw the pre-forming geometry. Each part is sketched as a 2D closed polygon and exported individually as a DXF file.

![Onshape DXF geometry — blank, punch, and die before forming](img/ss_onshape.png)
*Backward cup extrusion geometry drawn in Onshape (free). Left to right: blank (Ø30 × 20 mm billet), punch (Ø20 mm flat), die (container bore Ø30 mm). Each part is saved as a separate DXF file.*

Any CAD tool that exports DXF works — Onshape, FreeCAD, QCAD, AutoCAD, SolidWorks. The geometry just needs to be a closed 2D polygon.

### Step 2: Upload, configure, and run

![ForgeCal workflow — upload, configure, and run](img/ss_abcd_workflow.png)
*The complete ForgeCal workflow: (A) top page — no login required; (B) upload the three DXF files; (C) select material, friction model, and stroke; (D) the solver runs on our server in 30–90 seconds.*

Configure the simulation:

- **Material**: choose from built-in flow stress curves (S45C, SUJ2, SUS304, etc.) or enter your own piecewise linear data
- **Friction**: Coulomb or constant-shear (Tresca) model
- **Mode**: axisymmetric (round parts) or plane strain (long parts)

Hit **Run**. The solver runs on our server — your machine does nothing.

---

## What You Get for Free: The Load Curve

After the job completes, you get a **load-stroke curve** directly in the browser.

![Load-stroke result screen](img/ss_e_loadstroke.png)
*The punch force vs. stroke curve — the most critical output for press selection.*

This single graph answers the most important question in forging process design:

> **How much force does this operation require?**

From the peak load, you can:
- Select the right press tonnage
- Estimate die stress (by dividing load by contact area)
- Compare process variants (reduction ratio, corner radius, stroke length)
- Catch cases where the load exceeds your press capacity — *before* cutting steel

---

## Want More? Download the Results File

The analysis itself is completely free — run as many cases as you need, tweak parameters, compare variants, until you're satisfied with the result. Once you're happy, download the **results file (HDF5 format)** for deeper analysis.

```
Pricing model
──────────────────────────────────────────────────
Web solver (unlimited runs)        Free
Load-stroke curve in browser       Free
Results file (HDF5) download       Free now → Paid soon
Post-processor app (Windows EXE)   Free to download
Sample HDF5 file (GitHub)          Free
──────────────────────────────────────────────────
```

**Now is a good time to try it** — the HDF5 download is currently free while the service is in early access.

Once you have the results file, open it in the **free post-processor app**:

![Post-processor stress contour screen](img/ss_f_postprocessor.png)
*Stress and strain contour maps, deformed mesh animation, and detailed field data — all processed locally on your PC.*

The post-processor provides:

- von Mises stress contours at any stroke position
- Equivalent plastic strain (PEEQ) distribution
- Material flow visualization
- Frame-by-frame animation of the forming process

**Not ready to run a simulation yet?** A sample HDF5 file is included in the GitHub repository — download the post-processor and explore the interface with real data, no simulation required.

---

## Supported Processes

ForgeCal handles a wide range of cold forming operations in 2D:

```
Process                        Mode
────────────────────────────────────────
Backward cup extrusion         Axisymmetric
Forward extrusion              Axisymmetric
Closed-die forging             Axisymmetric
Multi-stage forging            Axisymmetric
V-bending / draw-bending       Plane strain
Blanking with ductile fracture Plane strain
```

If your part is round (axisymmetric) or long and uniform (plane strain), ForgeCal can model it.

---

## Limitations — Be Honest About What It Can't Do

ForgeCal is a **calculator**, not a replacement for full 3D simulation:

- **2D only**: no 3D effects, no asymmetric geometries
- **Isothermal**: no thermal softening — hot forging not supported
- **No self-contact**: folding defects cannot be modeled
- **Simplified fracture**: Cockcroft-Latham criterion only

For die stress analysis in 3D, fatigue life, or hot forging — you still need commercial software. ForgeCal fills the gap between hand calculation and full simulation.

---

## Try It

- **Web solver (free)**: https://fem.matsumoto-works.jp
- **Post-processor (free)**: download from the GitHub releases page (matsumoto-works/forgecal)
- **Sample results file**: included in the repository — open it in the post-processor without running anything
- **Benchmark model files**: included under `model_cases/`

The backward cup extrusion case used in the companion validation article is included — run it yourself and check the numbers.

---

*Keywords: metal forming, forging simulation, FEM, CAE, die design, open source, press selection*
