//
// Written by Nguyen Minh Hieu (Charlie), 2024
//
class ArcballCamera {
  constructor(gl, canvas, shaderProg, drawCallback) {
    this.gl     = gl;
    this.canvas = canvas;
    this.draw   = drawCallback;
    this.mvpLoc = gl.getUniformLocation(shaderProg, "u_mvp");
    this.eyeLoc = gl.getUniformLocation(shaderProg, "u_eye");
    this.timeLoc = gl.getUniformLocation(shaderProg, "u_time");

    this.world_mat = mat4.identity(mat4.create());

    this.eye      = vec3.fromValues(0.5, 0.5, 2.5);
    this.up       = vec3.fromValues(0.0, 1.0, 0.0);
    this.at       = vec3.fromValues(0.0, 0.0, 0.0);
    this.view_mat = new Float32Array(16);

    this.fov      = 45;
    this.near     = 0.1;
    this.far      = 1000;
    this.proj_mat = mat4.create();
    this.updateViewModel();
    this.updateProjection();
    this.mvp = this.getMVP();

    this.mouse_p  = vec2.create();
    this.mouse_dp = vec2.create();
    this.setInputHandler();

    document.body.style.overflow = 'hidden'; // disable scrolling
  }

  getMVP() {
    let mvp = mat4.create();
    mat4.mul(mvp, this.proj_mat, this.view_mat);
    mat4.mul(mvp, mvp, this.world_mat);
    return mvp;
  }

  updateEye() {
    let inv_view = mat4.create();
    mat4.mul(inv_view, this.view_mat, this.world_mat);
    mat4.invert(inv_view, inv_view);

    let new_eye = vec3.create();
    vec3.transformMat4(new_eye, vec3.fromValues(0,0,0), inv_view);
    this.gl.uniform3fv(this.eyeLoc, new_eye);
  }

  updateViewModel() {
    mat4.lookAt(this.view_mat, this.eye, this.at, this.up);
  }

  updateProjection() {
    let aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
    mat4.perspective(this.proj_mat, this.fov, aspectRatio, this.near, this.far);
  }

  getNDC(p) {
    let x = (2 * p[0] / this.canvas.width) - 1;
    let y = 1 - (2 * p[1] / this.canvas.height);
    let z = (x*x + y*y) <= 1 ? Math.sqrt(1 - x*x - y*y) : 0;
    return vec3.fromValues(x, y, z);
  }

  setInputHandler() {
    window.onresize = (e) => {
      if (document.hidden) return;
      this.canvas.width  = window.innerWidth;
      this.canvas.height = window.innerHeight;
      this.update();
    }

    window.onmousemove = (e) => {
      if (e.buttons === 1) {
        if (document.hidden) return;
        let curr_p_ndc = this.getNDC(vec2.fromValues(e.clientX, e.clientY));
        let prev_p_ndc = this.getNDC(vec2.fromValues(e.clientX-e.movementX, e.clientY-e.movementY));
        let angle = 2 * Math.acos(vec3.dot(curr_p_ndc, prev_p_ndc) / (vec3.length(curr_p_ndc) * vec3.length(prev_p_ndc)));
        let axis = vec3.create();
        vec3.cross(axis, prev_p_ndc, curr_p_ndc);
        let rot = mat4.fromRotation(mat4.create(), angle, axis);
        if (rot !== null) {
          this.world_mat = mat4.mul(this.world_mat, rot, this.world_mat);
          this.update()
        }
      }
    }

    window.onwheel = (e) => {
      if (document.hidden) return;
      let zoom = e.deltaY > 0 ? 1.01 : 0.99;
      let scale = mat4.fromScaling(mat4.create(), vec3.fromValues(zoom, zoom, zoom));
      this.world_mat = mat4.mul(this.world_mat, scale, this.world_mat);
      this.update();
    }
  }

  update() {
    this.gl.uniform1f(this.timeLoc, performance.now());
    this.updateEye();
    this.mvp = this.getMVP();
    this.gl.uniformMatrix4fv(this.mvpLoc, this.gl.FALSE, this.mvp);
  }
}
