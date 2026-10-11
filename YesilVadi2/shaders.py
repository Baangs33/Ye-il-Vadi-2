"""GLSL shader'ları: fiziksel tabanlı ışığa yakın aydınlatma (doğrusal renk uzayı, ACES ton eşleme),
yumuşak gölgeler, kabartma (bump) eşleme, yaprak yarı saydamlığı, atmosferik sis,
arazi doku karışımı, su, gökyüzü, inşaat planı (hayalet) ve ekran sonrası işleme (FXAA, bloom, AO, vinyet)."""
from ursina import Shader

LIGHT_STRUCT = '''
uniform struct {
    vec4 position;
    vec3 color;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];
'''

COMMON_FRAG = '''
uniform vec3 sun_dir;
uniform vec3 sun_col;
uniform vec3 amb_sky;
uniform vec3 amb_ground;
uniform vec3 fog_col;
uniform vec3 sun_glow;
uniform float fog_density;
uniform vec3 cam_pos;
uniform float exposure;
uniform float shadow_texel;
uniform vec4 lamps[8];
uniform vec3 lamp_col;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// Mikkelsen: uv gerektirmeyen yüzey gradyanı kabartması
vec3 bump_normal(vec3 N, vec3 P, float h, float k) {
    vec3 dpdx = dFdx(P), dpdy = dFdy(P);
    vec3 r1 = cross(dpdy, N), r2 = cross(N, dpdx);
    float det = dot(dpdx, r1);
    float dhx = dFdx(h), dhy = dFdy(h);
    vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
    return normalize(abs(det) * N - grad * k);
}

vec3 point_lights(vec3 P, vec3 N) {
    vec3 pl = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        if (lamps[i].w <= 0.001) continue;
        vec3 Lv = lamps[i].xyz - P;
        float d2 = dot(Lv, Lv);
        float att = lamps[i].w / (1.0 + d2 * 0.25) * (1.0 - smoothstep(30.0, 140.0, d2));
        float nl = clamp(dot(N, normalize(Lv)) * 0.8 + 0.2, 0.0, 1.0);
        pl += vec3(att * nl);
    }
    return pl * lamp_col;
}

vec3 finish(vec3 lin, vec3 P) {
    vec3 col = aces(lin * exposure);
    col = pow(col, vec3(1.0 / 2.2));
    vec3 V = P - cam_pos;
    float d = length(V);
    float f = 1.0 - exp(-pow(d * fog_density, 1.35));
    f *= 1.0 - 0.45 * clamp((P.y - 8.0) / 60.0, 0.0, 1.0);
    vec3 fc = fog_col + sun_glow * pow(max(dot(V / max(d, 1e-3), sun_dir), 0.0), 6.0) * 0.45;
    return mix(col, fc, clamp(f, 0.0, 1.0));
}
'''


def _shadow_fn(enabled):
    if not enabled:
        return 'float shadow_at(vec4 sc, vec3 N) { return 1.0; }\n'
    return '''
float shadow_at(vec4 sc, vec3 N) {
    vec3 c = sc.xyz / sc.w;
    if (c.x <= 0.001 || c.x >= 0.999 || c.y <= 0.001 || c.y >= 0.999 || c.z >= 1.0) return 1.0;
    float s = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++) {
            vec4 q = sc;
            q.xy += vec2(x, y) * shadow_texel * q.w;
            s += textureProj(p3d_LightSource[0].shadowMap, q);
        }
    float edge = smoothstep(0.0, 0.08, min(min(c.x, 1.0 - c.x), min(c.y, 1.0 - c.y)));
    return mix(1.0, s / 9.0, edge);
}
'''


def object_shader(shadows=True, name='obj'):
    vert = '''#version 150
''' + (LIGHT_STRUCT if shadows else '') + '''
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
uniform float osg_FrameTime;
uniform float wind_strength;

in vec4 vertex;
in vec3 normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
in vec4 params;

out vec4 v_color;
out vec3 v_pos;
out vec3 v_nrm;
out vec4 v_shadow;
out vec2 v_uv;
out vec4 v_params;

void main() {
    vec4 v = vertex;
    vec4 wp0 = p3d_ModelMatrix * vertex;
    float sway = params.x;
    if (sway > 0.001) {
        float t = osg_FrameTime;
        float ph = wp0.x * 0.21 + wp0.z * 0.17;
        float gust = 0.55 + 0.45 * sin(t * 0.43 + wp0.x * 0.013 + wp0.z * 0.009);
        vec3 off = vec3(sin(t * 1.7 + ph) + 0.4 * sin(t * 3.7 + ph * 2.3),
                        0.0,
                        0.6 * cos(t * 1.3 + ph * 0.7) + 0.3 * sin(t * 4.1 + ph)) * 0.16 * wind_strength * gust * sway;
        v.xyz += off;
    }
    gl_Position = p3d_ModelViewProjectionMatrix * v;
    v_pos = (p3d_ModelMatrix * v).xyz;
    v_nrm = normalize(mat3(p3d_ModelMatrix) * normal);
''' + ('''
    vec4 vpos = p3d_ModelViewMatrix * v;
    vpos.xyz += normalize(p3d_NormalMatrix * normal) * 0.06;
    v_shadow = p3d_LightSource[0].shadowViewMatrix * vpos;
''' if shadows else '    v_shadow = vec4(0.0);\n') + '''
    v_color = p3d_Color;
    v_uv = p3d_MultiTexCoord0;
    v_params = params;
}
'''
    frag = '''#version 150
''' + (LIGHT_STRUCT if shadows else '') + COMMON_FRAG + _shadow_fn(shadows) + '''
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform float emissive;
uniform float wetness;

in vec4 v_color;
in vec3 v_pos;
in vec3 v_nrm;
in vec4 v_shadow;
in vec2 v_uv;
in vec4 v_params;
out vec4 frag;

void main() {
    vec4 tex = texture(p3d_Texture0, v_uv);
    if (tex.a < 0.42) discard;
    vec3 base = tex.rgb * v_color.rgb * p3d_ColorScale.rgb;
    vec3 albedo = pow(base, vec3(2.2));
    vec3 N = normalize(v_nrm);
    if (v_params.z > 0.0) N = bump_normal(N, v_pos, luma(tex.rgb), v_params.z);
    vec3 L = sun_dir;
    float ndl = dot(N, L);
    float sh = shadow_at(v_shadow, N);
    vec3 V = normalize(cam_pos - v_pos);
    vec3 diff = sun_col * clamp(ndl, 0.0, 1.0) * sh;
    vec3 amb = mix(amb_ground, amb_sky, N.y * 0.5 + 0.5) * v_color.a;
    vec3 trans = sun_col * v_params.w * pow(clamp(dot(-V, L), 0.0, 1.0), 2.0) * (0.35 + 0.65 * sh) * 0.9
               + sun_col * v_params.w * clamp(-ndl, 0.0, 1.0) * 0.25 * sh;
    vec3 col = albedo * (diff + amb + trans);
    float spec_k = v_params.y + wetness * 0.4;
    if (spec_k > 0.0) {
        vec3 H = normalize(V + L);
        float fres = 0.04 + 0.96 * pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 5.0);
        float gloss = 16.0 + spec_k * 180.0;
        col += sun_col * pow(clamp(dot(N, H), 0.0, 1.0), gloss) * (gloss + 8.0) / 25.0 * spec_k * 0.35 * sh;
        col += mix(amb_ground, amb_sky, reflect(-V, N).y * 0.5 + 0.5) * fres * spec_k * 0.8;
    }
    col += albedo * point_lights(v_pos, N);
    col += albedo * emissive * 4.0;
    frag = vec4(finish(col, v_pos), 1.0);
}
'''
    return Shader(language=Shader.GLSL, name=name, vertex=vert, fragment=frag,
                  default_input={'emissive': 0.0, 'wetness': 0.0})


world_shader = object_shader(True, 'vadi_obj')
viewmodel_shader = object_shader(False, 'vadi_vm')


terrain_shader = Shader(language=Shader.GLSL, name='vadi_terrain', vertex='''#version 150
''' + LIGHT_STRUCT + '''
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
in vec4 vertex;
in vec3 normal;
in vec4 p3d_Color;
out vec4 v_w;
out vec3 v_pos;
out vec3 v_nrm;
out vec4 v_shadow;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    v_pos = (p3d_ModelMatrix * vertex).xyz;
    v_nrm = normalize(mat3(p3d_ModelMatrix) * normal);
    vec4 vpos = p3d_ModelViewMatrix * vertex;
    vpos.xyz += normalize(p3d_NormalMatrix * normal) * 0.08;
    v_shadow = p3d_LightSource[0].shadowViewMatrix * vpos;
    v_w = p3d_Color;
}
''', fragment='''#version 150
''' + LIGHT_STRUCT + COMMON_FRAG + _shadow_fn(True) + '''
uniform sampler2D tex_grass;
uniform sampler2D tex_dirt;
uniform sampler2D tex_rock;
uniform sampler2D tex_sand;
uniform sampler2D tex_noise;
uniform float wetness;
in vec4 v_w;
in vec3 v_pos;
in vec3 v_nrm;
in vec4 v_shadow;
out vec4 frag;

vec3 sample2(sampler2D t, vec2 p, float s) {
    vec3 a = texture(t, p / s).rgb;
    vec3 b = texture(t, p / (s * 3.7) + vec2(0.37, 0.71)).rgb;
    return mix(a, b, 0.35);
}

void main() {
    vec3 N = normalize(v_nrm);
    vec2 p = v_pos.xz;
    vec3 mac = texture(tex_noise, p / 90.0).rgb;
    vec3 g = sample2(tex_grass, p, 3.2) * (0.80 + 0.45 * mac.r);
    g = mix(g, g * vec3(1.15, 1.05, 0.75), smoothstep(0.55, 0.75, mac.g) * 0.6);
    vec3 d = sample2(tex_dirt, p, 2.6);
    vec3 s = sample2(tex_sand, p, 3.0);
    vec3 an = abs(N);
    an = pow(an, vec3(4.0)); an /= (an.x + an.y + an.z);
    vec3 r = (texture(tex_rock, v_pos.zy / 7.0).rgb * an.x + texture(tex_rock, v_pos.xz / 7.0).rgb * an.y
            + texture(tex_rock, v_pos.xy / 7.0).rgb * an.z) * (0.85 + 0.3 * mac.b);
    vec4 w = v_w;
    // yükseklik tabanlı keskin geçişler
    vec4 hts = vec4(luma(g) + 0.25, luma(d) + 0.1, luma(r) + 0.45, luma(s));
    w *= pow(hts + 0.5, vec4(6.0));
    w /= max(w.x + w.y + w.z + w.w, 1e-4);
    vec3 base = g * w.x + d * w.y + r * w.z + s * w.w;
    float h = luma(base);
    N = bump_normal(N, v_pos, h, 0.12 + w.z * 0.9);
    vec3 albedo = pow(base, vec3(2.2));
    albedo *= 1.0 - wetness * 0.35 * (w.y + w.w);
    float sh = shadow_at(v_shadow, N);
    float ndl = clamp(dot(N, sun_dir), 0.0, 1.0);
    vec3 col = albedo * (sun_col * ndl * sh + mix(amb_ground, amb_sky, N.y * 0.5 + 0.5));
    vec3 V = normalize(cam_pos - v_pos);
    float spec_k = 0.05 + wetness * 0.5 * (w.y + w.w + w.z * 0.5);
    vec3 H = normalize(V + sun_dir);
    col += sun_col * pow(clamp(dot(N, H), 0.0, 1.0), 40.0 + wetness * 200.0) * spec_k * sh;
    col += albedo * point_lights(v_pos, N);
    frag = vec4(finish(col, v_pos), 1.0);
}
''', default_input={'wetness': 0.0})


sky_shader = Shader(language=Shader.GLSL, name='vadi_sky', vertex='''#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 vertex;
out vec3 v_dir;
void main() { v_dir = vertex.xyz; gl_Position = p3d_ModelViewProjectionMatrix * vertex; }
''', fragment='''#version 150
uniform vec3 sun_dir;
uniform vec3 sky_top;
uniform vec3 sky_hor;
uniform vec3 sun_glow;
uniform vec3 cloud_lit;
uniform vec3 cloud_dark;
uniform float night;
uniform float cloud_cover;
uniform float osg_FrameTime;
in vec3 v_dir;
out vec4 frag;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash13(vec3 p3){ p3 = fract(p3 * .1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 6; i++){ s += a * vnoise(p); p = p * 2.02 + vec2(1.7, 9.2); a *= 0.5; } return s; }
void main() {
    vec3 d = normalize(v_dir);
    float y = d.y;
    float t = osg_FrameTime;
    float hz = pow(1.0 - clamp(y, 0.0, 1.0), 4.0);
    vec3 col = mix(sky_top, sky_hor, hz);
    if (y < 0.0) col = mix(sky_hor, sky_hor * 0.75, clamp(-y * 3.0, 0.0, 1.0));
    float sd = max(dot(d, sun_dir), 0.0);
    col += sun_glow * (pow(sd, 4.0) * 0.25 + pow(sd, 32.0) * 0.45 + pow(sd, 300.0) * 1.2);
    vec3 cell = floor(d * 260.0);
    float hs = hash13(cell);
    float star = step(0.9968, hs) * (0.5 + 0.5 * sin(t * 2.5 + hs * 300.0));
    col += vec3(star) * night * smoothstep(0.02, 0.3, y);
    vec3 md = -sun_dir;
    float mdd = max(dot(d, md), 0.0);
    float moon = smoothstep(0.99930, 0.99955, mdd);
    col += vec3(0.25, 0.30, 0.45) * pow(mdd, 25.0) * 0.4 * night;
    float disk = smoothstep(0.99965, 0.99982, sd);
    float cloud = 0.0;
    vec3 ccol = vec3(0.0);
    if (y > -0.02) {
        vec2 uv = d.xz / (y + 0.08) * 0.75 + vec2(t * 0.006, t * 0.0025);
        float n = fbm(uv);
        float thr = mix(0.58, 0.30, cloud_cover);
        cloud = smoothstep(thr, thr + 0.28, n) * smoothstep(-0.02, 0.2, y);
        float n2 = fbm(uv + normalize(sun_dir.xz + 1e-4) * 0.07);
        float lit = clamp((n - n2) * 4.5 + 0.55, 0.0, 1.0);
        ccol = mix(cloud_dark, cloud_lit, lit);
        ccol += sun_glow * pow(sd, 6.0) * 0.5 * (1.0 - cloud * 0.5);
    }
    col = mix(col, vec3(1.0, 0.97, 0.88) * 1.6, disk * (1.0 - cloud));
    col = mix(col, vec3(0.92, 0.94, 1.0), moon * night * (1.0 - cloud));
    col = mix(col, ccol, cloud * 0.95);
    frag = vec4(col, 1.0);
}
''')


water_shader = Shader(language=Shader.GLSL, name='vadi_water', vertex='''#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform float osg_FrameTime;
in vec4 vertex;
in vec2 p3d_MultiTexCoord0;
out vec3 v_pos;
out float v_depth;
void main() {
    vec4 v = vertex;
    vec4 wp = p3d_ModelMatrix * v;
    float t = osg_FrameTime;
    v.y += (sin(wp.x * 0.35 + t * 1.1) + cos(wp.z * 0.4 + t * 0.9)) * 0.035;
    gl_Position = p3d_ModelViewProjectionMatrix * v;
    v_pos = (p3d_ModelMatrix * v).xyz;
    v_depth = p3d_MultiTexCoord0.x;
}
''', fragment='''#version 150
uniform vec3 sun_dir;
uniform vec3 sun_col;
uniform vec3 sky_top;
uniform vec3 fog_col;
uniform vec3 sun_glow;
uniform float fog_density;
uniform vec3 cam_pos;
uniform float osg_FrameTime;
uniform sampler2D tex_noise;
in vec3 v_pos;
in float v_depth;
out vec4 frag;
void main() {
    float t = osg_FrameTime;
    vec2 p = v_pos.xz;
    vec2 o1 = vec2(t * 0.018, t * 0.011), o2 = vec2(-t * 0.013, t * 0.017);
    vec3 n1 = texture(tex_noise, p / 9.0 + o1).rgb;
    vec3 n2 = texture(tex_noise, p / 4.3 + o2).rgb;
    vec3 n3 = texture(tex_noise, p / 1.7 - o1 * 2.0).rgb;
    vec2 g = (n1.rg - 0.5) * 0.9 + (n2.gb - 0.5) * 0.6 + (n3.rb - 0.5) * 0.35;
    vec3 N = normalize(vec3(g.x, 1.0, g.y));
    vec3 V = normalize(cam_pos - v_pos);
    float fres = 0.03 + 0.97 * pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 5.0);
    float dpt = clamp(v_depth, 0.0, 6.0);
    vec3 shallow = vec3(0.10, 0.30, 0.28);
    vec3 deep = vec3(0.015, 0.07, 0.09);
    float daylight = clamp(sun_dir.y * 2.0 + 0.2, 0.08, 1.0);
    vec3 body = mix(shallow, deep, smoothstep(0.0, 3.5, dpt)) * daylight;
    vec3 refl = mix(fog_col, sky_top, 0.45 + 0.4 * clamp(reflect(-V, N).y, 0.0, 1.0));
    vec3 col = mix(body, refl, fres);
    vec3 R = reflect(-sun_dir, N);
    col += sun_col * pow(max(dot(R, V), 0.0), 220.0) * 0.6;
    float foam = (1.0 - smoothstep(0.0, 0.35, dpt)) * (0.5 + 0.5 * sin(t * 1.5 + n1.r * 12.0));
    col = mix(col, vec3(0.85, 0.9, 0.9) * daylight, foam * 0.35);
    float alpha = mix(0.35, 0.95, smoothstep(0.0, 1.8, dpt));
    alpha = max(alpha, fres);
    float d = length(v_pos - cam_pos);
    float f = 1.0 - exp(-pow(d * fog_density, 1.35));
    col = mix(col, fog_col, f);
    frag = vec4(col, alpha);
}
''')


ghost_shader = Shader(language=Shader.GLSL, name='vadi_ghost', vertex='''#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 vertex;
in vec3 normal;
out vec3 v_pos;
out vec3 v_nrm;
void main() { gl_Position = p3d_ModelViewProjectionMatrix * vertex; v_pos = (p3d_ModelMatrix * vertex).xyz; v_nrm = normalize(mat3(p3d_ModelMatrix) * normal); }
''', fragment='''#version 150
uniform vec4 ghost_col;
uniform vec3 cam_pos;
uniform float osg_FrameTime;
in vec3 v_pos;
in vec3 v_nrm;
out vec4 frag;
void main() {
    vec3 V = normalize(cam_pos - v_pos);
    float rim = pow(1.0 - abs(dot(normalize(v_nrm), V)), 2.0);
    float scan = 0.5 + 0.5 * sin(v_pos.y * 9.0 - osg_FrameTime * 3.0);
    vec3 c = ghost_col.rgb * (0.55 + 0.6 * rim + 0.15 * scan);
    frag = vec4(c, ghost_col.a * (0.35 + 0.65 * rim));
}
''', default_input={'ghost_col': (0.55, 0.8, 1.0, 0.45)})


glow_shader = Shader(language=Shader.GLSL, name='vadi_glow', vertex='''#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 vertex;
in vec4 p3d_Color;
out vec4 v_color;
void main() { gl_Position = p3d_ModelViewProjectionMatrix * vertex; v_color = p3d_Color; }
''', fragment='''#version 150
uniform vec4 p3d_ColorScale;
uniform float glow;
in vec4 v_color;
out vec4 frag;
void main() { vec4 c = v_color * p3d_ColorScale; frag = vec4(c.rgb * glow, c.a); }
''', default_input={'glow': 1.0})


rain_shader = Shader(language=Shader.GLSL, name='vadi_rain', vertex='''#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform float osg_FrameTime;
in vec4 vertex;
in vec4 p3d_Color;
out float v_a;
void main() {
    vec4 v = vertex;
    float off = p3d_Color.r * 40.0;
    v.y = mod(v.y - osg_FrameTime * 22.0 + off, 24.0) - 8.0;
    v.x += v.y * 0.06;
    gl_Position = p3d_ModelViewProjectionMatrix * v;
    v_a = p3d_Color.g;
}
''', fragment='''#version 150
uniform float rain_alpha;
uniform vec3 rain_col;
in float v_a;
out vec4 frag;
void main() { frag = vec4(rain_col, rain_alpha * v_a); }
''', default_input={'rain_alpha': 0.0, 'rain_col': (0.7, 0.75, 0.85)})


post_shader = Shader(language=Shader.GLSL, name='vadi_post', vertex='''#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;
void main() { gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; uv = p3d_MultiTexCoord0; }
''', fragment='''#version 140
uniform sampler2D tex;
uniform sampler2D dtex;
uniform vec2 texel;
uniform float bloom_amt;
uniform float ao_amt;
uniform float vignette;
uniform float saturation;
uniform float contrast;
uniform float fxaa_on;
uniform float near;
uniform float far;
uniform float damage;
in vec2 uv;
out vec4 o_color;

float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float lin_depth(vec2 p) { float z = texture(dtex, p).r; return near * far / (far - z * (far - near)); }

vec3 fxaa(vec2 p) {
    vec3 nw = texture(tex, p + vec2(-1.0, -1.0) * texel).rgb;
    vec3 ne = texture(tex, p + vec2(1.0, -1.0) * texel).rgb;
    vec3 sw = texture(tex, p + vec2(-1.0, 1.0) * texel).rgb;
    vec3 se = texture(tex, p + vec2(1.0, 1.0) * texel).rgb;
    vec3 m = texture(tex, p).rgb;
    float lnw = lum(nw), lne = lum(ne), lsw = lum(sw), lse = lum(se), lm = lum(m);
    float lmin = min(lm, min(min(lnw, lne), min(lsw, lse)));
    float lmax = max(lm, max(max(lnw, lne), max(lsw, lse)));
    vec2 dir = vec2(-((lnw + lne) - (lsw + lse)), ((lnw + lsw) - (lne + lse)));
    float red = max((lnw + lne + lsw + lse) * 0.25 * (1.0 / 8.0), 1.0 / 128.0);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + red);
    dir = clamp(dir * rcp, vec2(-8.0), vec2(8.0)) * texel;
    vec3 a = 0.5 * (texture(tex, p + dir * (1.0 / 3.0 - 0.5)).rgb + texture(tex, p + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(tex, p + dir * -0.5).rgb + texture(tex, p + dir * 0.5).rgb);
    float lb = lum(b);
    return (lb < lmin || lb > lmax) ? a : b;
}

void main() {
    vec3 col = fxaa_on > 0.5 ? fxaa(uv) : texture(tex, uv).rgb;
    if (ao_amt > 0.0) {
        float d0 = lin_depth(uv);
        float occ = 0.0;
        float rad = clamp(1.6 / d0, 0.002, 0.03);
        for (int i = 0; i < 12; i++) {
            float a = float(i) * 2.39996;
            float r = rad * (0.3 + 0.7 * fract(float(i) * 0.618));
            vec2 o = vec2(cos(a), sin(a)) * r * vec2(texel.y / texel.x, 1.0);
            float ds = lin_depth(uv + o);
            float diff = d0 - ds;
            occ += smoothstep(0.05, 0.5, diff) * (1.0 - smoothstep(1.0, 4.0, diff));
        }
        occ = occ / 12.0 * (1.0 - smoothstep(60.0, 150.0, d0));
        col *= 1.0 - occ * ao_amt;
    }
    if (bloom_amt > 0.0) {
        vec3 b = vec3(0.0);
        for (int i = 0; i < 16; i++) {
            float a = float(i) * 2.39996;
            float r = (4.0 + float(i) * 1.6);
            vec2 o = vec2(cos(a), sin(a)) * r * texel;
            vec3 s = texture(tex, uv + o).rgb;
            b += max(s - 0.72, 0.0);
        }
        col += b / 16.0 * bloom_amt * 2.2;
    }
    float l = lum(col);
    col = mix(vec3(l), col, saturation);
    col = (col - 0.5) * contrast + 0.5;
    col *= vec3(1.02, 1.0, 0.97);
    vec2 q = uv - 0.5;
    col *= 1.0 - dot(q, q) * vignette;
    col = mix(col, vec3(0.6, 0.05, 0.02), damage * smoothstep(0.15, 0.7, length(q)));
    o_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', default_input={'bloom_amt': 0.5, 'ao_amt': 0.0, 'vignette': 0.9, 'saturation': 1.0, 'contrast': 1.05,
                    'fxaa_on': 1.0, 'near': 0.1, 'far': 900.0, 'damage': 0.0, 'texel': (1 / 1280, 1 / 720)})
