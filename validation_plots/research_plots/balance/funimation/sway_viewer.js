import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class SwayViewer {
  constructor(opts) {
    const defaults = {
      sceneEl: '#scene',
      dataUrl: '/sway.json',
      leftKey: null,
      rightKey: null,
    };
    this.opts = Object.assign({}, defaults, opts || {});
    this.container = this._elt(this.opts.sceneEl);

    this.k = 0;
    this.playing = false;
    this.lastT = 0;

    this._initThree();
    this._wireUI();
    this._loadData();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  _elt(sel) {
    const el = typeof sel === 'string' ? document.querySelector(sel) : sel;
    if (!el) throw new Error(`Element not found: ${sel}`);
    return el;
  }

  _initThree() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.container.appendChild(this.renderer.domElement);

    // Two scenes (side-by-side)
    this.sceneL = new THREE.Scene();
    this.sceneR = new THREE.Scene();
    this.sceneL.background = new THREE.Color(0x0b0e14);
    this.sceneR.background = new THREE.Color(0x0b0e14);

    this.cameraL = new THREE.PerspectiveCamera(55, 1, 0.1, 20000);
    this.cameraR = new THREE.PerspectiveCamera(55, 1, 0.1, 20000);

    this.cameraL.up.set(0, 0, 1);
    this.cameraR.up.set(0, 0, 1);

    this.controlsL = new OrbitControls(this.cameraL, this.renderer.domElement);
    this.controlsR = new OrbitControls(this.cameraR, this.renderer.domElement);
    this.controlsL.enableDamping = true;
    this.controlsR.enableDamping = true;

    const addLights = (scene) => {
      scene.add(new THREE.AmbientLight(0xffffff, 0.35));
      const d = new THREE.DirectionalLight(0xffffff, 1.0);
      d.position.set(1, -1, 2);
      scene.add(d);
      const grid = new THREE.GridHelper(240, 12, 0x22304d, 0x182238);
      grid.rotation.x = Math.PI / 2; // ML/AP plane
      grid.material.opacity = 0.5;
      grid.material.transparent = true;
      scene.add(grid);
      scene.add(new THREE.AxesHelper(60));
    };
    addLights(this.sceneL);
    addLights(this.sceneR);

    // resize
    this.renderer.setScissorTest(true);
    const resize = () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.renderer.setSize(w, h);
    };
    new ResizeObserver(resize).observe(this.container);
    resize();
  }

  _wireUI() {
    this.ui = {
      playBtn: document.getElementById('playBtn'),
      speed: document.getElementById('speed'),
      speedVal: document.getElementById('speedVal'),
      tail: document.getElementById('tail'),
      tailVal: document.getElementById('tailVal'),
      scrub: document.getElementById('scrub'),
      frameVal: document.getElementById('frameVal'),
      meta: document.getElementById('meta'),
    };

    this.ui.playBtn.addEventListener('click', () => {
      this.playing = !this.playing;
      this.ui.playBtn.textContent = this.playing ? 'Pause' : 'Play';
      this.lastT = 0;
    });

    const upd = () => {
      this.ui.speedVal.textContent = `${Number(this.ui.speed.value).toFixed(2)}×`;
      this.ui.tailVal.textContent = `${this.ui.tail.value}`;
    };
    this.ui.speed.addEventListener('input', upd);
    this.ui.tail.addEventListener('input', upd);
    upd();

    this.ui.scrub.addEventListener('input', () => {
      this.setFrame(Number(this.ui.scrub.value));
    });

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        this.ui.playBtn.click();
      }
    });
  }

  async _loadData() {
    const res = await fetch(this.opts.dataUrl, { cache: 'no-store' });
    const json = await res.json();

    this.data = json;
    const keys = Object.keys(json.conditions || {});
    if (keys.length < 1) throw new Error('No conditions found in sway.json');

    this.leftKey = this.opts.leftKey || keys[0];
    this.rightKey = this.opts.rightKey || (keys[1] || keys[0]);

    const NL = json.conditions[this.leftKey].ml.length;
    const NR = json.conditions[this.rightKey].ml.length;
    this.F = Math.min(NL, NR);

    this.ui.scrub.max = String(this.F - 1);
    this.ui.meta.textContent = `${json.trial_name} • ${json.tracker}`;

    // build actors
    this.actorL = this._makeActor(this.sceneL, json.conditions[this.leftKey], this.leftKey);
    this.actorR = this._makeActor(this.sceneR, json.conditions[this.rightKey], this.rightKey);

    // camera framing
    this._frameCamera(this.cameraL, this.controlsL, json.conditions[this.leftKey]);
    this._frameCamera(this.cameraR, this.controlsR, json.conditions[this.rightKey]);

    this.setFrame(0);
  }

  _makeActor(scene, d, title) {
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(2.6, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x111111, roughness: 0.35, metalness: 0.2 })
    );
    scene.add(sphere);

    const maxTrail = 800;
    const positions = new Float32Array(maxTrail * 3);
    const colors = new Float32Array(maxTrail * 3);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.setDrawRange(0, 0);

    const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 });
    const line = new THREE.Line(geom, mat);
    scene.add(line);

    const vel = d.vel || [];
    const vmin = this._percentile(vel, 5);
    const vmax = this._percentile(vel, 95);

    const velToRGB = (v) => {
      const t = this._clamp((v - vmin) / (vmax - vmin + 1e-9), 0, 1);
      // blue -> red ramp
      return [t, 0.25 + 0.15*(1-t), 1 - t];
    };

    return { d, sphere, geom, velToRGB, maxTrail, title };
  }

  _frameCamera(camera, controls, d) {
    const ml = d.ml, ap = d.ap, z = d.z;
    const min = new THREE.Vector3(Math.min(...ml), Math.min(...ap), Math.min(...z));
    const max = new THREE.Vector3(Math.max(...ml), Math.max(...ap), Math.max(...z));
    const center = min.clone().add(max).multiplyScalar(0.5);
    const size = max.clone().sub(min);
    const radius = Math.max(size.x, size.y, size.z) * 0.9 + 20;

    camera.position.set(center.x, center.y - radius, center.z + radius * 0.55);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
  }

  setFrame(k) {
    if (!this.data || !this.F) return;
    this.k = Math.max(0, Math.min(this.F - 1, k));
    this.ui.scrub.value = String(this.k);
    this.ui.frameVal.textContent = String(this.k);

    this._updateActor(this.actorL, this.k);
    this._updateActor(this.actorR, this.k);
  }

  _updateActor(actor, f) {
    const d = actor.d;

    actor.sphere.position.set(d.ml[f], d.ap[f], d.z[f]);

    const tailLen = Math.min(Number(this.ui.tail.value), actor.maxTrail, f + 1);
    const start = Math.max(0, f - tailLen + 1);

    const posAttr = actor.geom.getAttribute('position');
    const colAttr = actor.geom.getAttribute('color');

    let j = 0;
    for (let i = start; i <= f; i++) {
      posAttr.setXYZ(j, d.ml[i], d.ap[i], d.z[i]);
      const [r,g,b] = actor.velToRGB((d.vel && d.vel[i]) ? d.vel[i] : 0);
      colAttr.setXYZ(j, r, g, b);
      j++;
    }

    actor.geom.setDrawRange(0, j);
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  }

  _animate(t) {
    // playback timing
    if (this.playing && this.F) {
      const fpsBase = 60;
      const speed = Math.max(0.25, Number(this.ui.speed.value) || 1);
      const frameTime = 1000 / (fpsBase * speed);

      if (!this.lastT) this.lastT = t;
      if (t - this.lastT >= frameTime) {
        this.lastT += frameTime;
        const next = this.k + 1;
        if (next >= this.F) {
          this.playing = false;
          this.ui.playBtn.textContent = 'Play';
        } else {
          this.setFrame(next);
        }
      }
    }

    this.controlsL.update();
    this.controlsR.update();
    this._renderSplit();
    requestAnimationFrame(this._animate);
  }

  _renderSplit() {
    const w = this.renderer.domElement.clientWidth;
    const h = this.renderer.domElement.clientHeight;
    const half = Math.floor(w / 2);

    // left
    this.renderer.setViewport(0, 0, half, h);
    this.renderer.setScissor(0, 0, half, h);
    this.cameraL.aspect = half / h;
    this.cameraL.updateProjectionMatrix();
    this.renderer.render(this.sceneL, this.cameraL);

    // right
    this.renderer.setViewport(half, 0, w - half, h);
    this.renderer.setScissor(half, 0, w - half, h);
    this.cameraR.aspect = (w - half) / h;
    this.cameraR.updateProjectionMatrix();
    this.renderer.render(this.sceneR, this.cameraR);
  }

  _clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  _percentile(arr, p) {
    const a = Array.from(arr || []).filter(v => typeof v === 'number').sort((x,y)=>x-y);
    if (!a.length) return 0;
    const idx = (p/100) * (a.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return a[lo];
    const t = idx - lo;
    return a[lo]*(1-t) + a[hi]*t;
  }
}
