// Núcleo de la perturbación, compartido entre la DLL de CPU (perturbacion.cpp) y el kernel
// de GPU (se compila con NVRTC desde core/funciones_kernel.py).
//
// Cuatro piezas:
//   1. Tipos numéricos: complejo double (CD) y complejo "floatexp" (CFE), un double con un
//      exponente entero aparte. El floatexp no tiene underflow, así que deja hacer zooms
//      mucho más allá de 1e-308.
//   2. Fórmulas: cada fractal (Mandelbrot, y a futuro Julia, Burning Ship...) es una estructura
//      que dice cómo se perturba, cómo se deriva y cómo se linealiza. Todo lo demás es genérico.
//   3. BLA (bilinear approximation): mientras dz es chico frente a Z, la iteración es casi
//      lineal, dz -> A dz + B dc, y varios pasos lineales se componen en uno solo. Se arma
//      una tabla de saltos de 1, 2, 4, 8... iteraciones y cada píxel usa el salto más largo
//      que siga siendo válido. Así se saltean miles de iteraciones de golpe.
//   4. La iteración de un píxel: BLA cuando se puede, perturbación normal cuando no,
//      rebasing para evitar glitches, y al final el valor suave, la distancia y la normal.
//
// Referencia: Claude Heiland-Allen, "Deep zoom theory and practice" (mathr.co.uk, 2021).
#pragma once

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
  #define HD __host__ __device__ __forceinline__
#else
  #include <cmath>
  #include <math.h>      // frexpf, ldexpf, fmaf... (versiones float)
  using std::frexp; using std::ldexp; using std::fabs; using std::fmax; using std::sqrt;
  using std::log; using std::log2; using std::fma;
  #define HD inline
#endif

// ---------------------------------------------------------------------------
// Tipos numéricos
// ---------------------------------------------------------------------------

struct CD  { double re, im; };          // complejo double
struct FE  { double m; int e; };        // real floatexp: m * 2^e, con 0.5 <= |m| < 1 (o m == 0)
struct CFE { double re, im; int e; };   // complejo floatexp: (re + i im) * 2^e, exponente compartido

// x - x es 0 salvo para inf y NaN
HD bool finito(double x) { return x - x == 0.0; }

// --- FE ---
HD FE fe_normalizar(double m, int e) {
    if (m == 0.0 || !finito(m)) { FE r = {m, 0}; return r; }
    int k; double mm = frexp(m, &k);
    FE r = {mm, e + k}; return r;
}
HD FE fe(double x) { return fe_normalizar(x, 0); }
HD FE operator*(FE a, FE b) { return fe_normalizar(a.m * b.m, a.e + b.e); }
HD FE operator/(FE a, FE b) { return fe_normalizar(a.m / b.m, a.e - b.e); }
HD FE operator+(FE a, FE b) {
    if (a.m == 0.0) return b;
    if (b.m == 0.0) return a;
    if (a.e >= b.e) return fe_normalizar(a.m + ldexp(b.m, b.e - a.e), a.e);
    return fe_normalizar(b.m + ldexp(a.m, a.e - b.e), b.e);
}
HD FE operator-(FE a, FE b) { b.m = -b.m; return a + b; }
HD double a_double(FE a) { return ldexp(a.m, a.e); }
HD double a_double(double a) { return a; }

// --- CD ---
HD CD operator+(CD a, CD b) { CD r = {a.re + b.re, a.im + b.im}; return r; }
// Las multiplicaciones usan FMA explícito: los compiladores tienen prohibido fusionar por su
// cuenta (-ffp-contract=off, --fmad=false), así CPU y GPU hacen exactamente las mismas
// operaciones y dan el mismo resultado bit a bit, sin perder la velocidad de las FMA.
HD CD operator*(CD a, CD b) {
    CD r = {fma(a.re, b.re, -(a.im * b.im)), fma(a.re, b.im, a.im * b.re)}; return r;
}
// a * b + c
HD CD mul_sum(CD a, CD b, CD c) {
    CD r = {fma(a.re, b.re, fma(-a.im, b.im, c.re)), fma(a.re, b.im, fma(a.im, b.re, c.im))}; return r;
}
HD CD operator-(CD a, CD b) { CD r = {a.re - b.re, a.im - b.im}; return r; }
HD CD doble(CD a) { CD r = {2.0 * a.re, 2.0 * a.im}; return r; }
HD CD conjugado(CD a) { a.im = -a.im; return a; }

// --- CFE ---
HD CFE cfe_normalizar(double re, double im, int e) {
    double m = fmax(fabs(re), fabs(im));
    if (m == 0.0 || !finito(m)) { CFE r = {re, im, 0}; return r; }
    int k; frexp(m, &k);
    CFE r = {ldexp(re, -k), ldexp(im, -k), e + k}; return r;
}
HD CFE operator+(CFE a, CFE b) {
    if (a.re == 0.0 && a.im == 0.0) return b;
    if (b.re == 0.0 && b.im == 0.0) return a;
    if (a.e >= b.e) {
        double s = ldexp(1.0, b.e - a.e);   // 0 si la diferencia de exponentes es enorme
        return cfe_normalizar(a.re + b.re * s, a.im + b.im * s, a.e);
    }
    double s = ldexp(1.0, a.e - b.e);
    return cfe_normalizar(b.re + a.re * s, b.im + a.im * s, b.e);
}
HD CFE operator*(CFE a, CFE b) {
    return cfe_normalizar(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re, a.e + b.e);
}
HD CFE operator-(CFE a, CFE b) { b.re = -b.re; b.im = -b.im; return a + b; }
HD CFE doble(CFE a) { a.e += 1; return a; }
HD CFE mul_sum(CFE a, CFE b, CFE c) { return a * b + c; }
HD CFE conjugado(CFE a) { a.im = -a.im; return a; }

// --- Operaciones genéricas: cada tipo complejo C tiene un tipo real R asociado ---
// norma2 = |z|^2, modulo = |z|, cota = sqrt(2) * max(|re|, |im|) >= |z| (barata y sin underflow)
HD double norma2(CD a) { return fma(a.re, a.re, a.im * a.im); }
HD double modulo(CD a) { return sqrt(norma2(a)); }
HD double cota(CD a)   { return 1.4142135623730951 * fmax(fabs(a.re), fabs(a.im)); }
HD bool   finito(CD a) { return finito(a.re) && finito(a.im); }

HD FE   norma2(CFE a) { return fe_normalizar(a.re * a.re + a.im * a.im, 2 * a.e); }
HD FE   modulo(CFE a) { return fe_normalizar(sqrt(a.re * a.re + a.im * a.im), a.e); }
HD FE   cota(CFE a)   { return fe_normalizar(1.4142135623730951 * fmax(fabs(a.re), fabs(a.im)), a.e); }
HD bool finito(CFE a) { return finito(a.re) && finito(a.im); }

// Dirección (vector unitario) de un complejo, sin overflow aunque sea enorme
HD void direccion(double re, double im, float* nx, float* ny) {
    double s = fmax(fabs(re), fabs(im));
    if (s == 0.0 || !finito(s)) { *nx = 0.0f; *ny = 0.0f; return; }
    re /= s; im /= s;
    double m = sqrt(re * re + im * im);
    *nx = (float)(re / m); *ny = (float)(im / m);
}
HD void direccion(CD a, float* nx, float* ny)  { direccion(a.re, a.im, nx, ny); }
HD void direccion(CFE a, float* nx, float* ny) { direccion(a.re, a.im, nx, ny); }

// Comparaciones. menor_pos asume a, b >= 0 (normas, radios): es la rápida, la que se usa por iteración.
HD bool menor_pos(double a, double b) { return a < b; }
HD bool menor_pos(FE a, FE b) {
    if (b.m == 0.0) return false;
    if (a.m == 0.0) return true;
    if (a.e != b.e) return a.e < b.e;
    return a.m < b.m;
}
HD bool positivo(double a) { return a > 0.0; }          // false también para NaN
HD bool positivo(FE a)     { return a.m > 0.0 && finito(a.m); }
HD bool es_cero(double a)  { return a == 0.0; }
HD bool es_cero(FE a)      { return a.m == 0.0; }

// Constructores genéricos
template<class C> HD C complejo(double re, double im);
template<> HD CD  complejo<CD>(double re, double im)  { CD r = {re, im}; return r; }
template<> HD CFE complejo<CFE>(double re, double im) { return cfe_normalizar(re, im, 0); }

template<class R> HD R real(double x);
template<> HD double real<double>(double x) { return x; }
template<> HD FE     real<FE>(double x)     { return fe(x); }

// dc a partir de sus partes en floatexp (así llegan desde Python, sin underflow)
template<class C> HD C complejo_fe(FE re, FE im);
template<> HD CD complejo_fe<CD>(FE re, FE im) { CD r = {a_double(re), a_double(im)}; return r; }
template<> HD CFE complejo_fe<CFE>(FE re, FE im) {
    int e = re.e > im.e ? re.e : im.e;
    if (re.m == 0.0) e = im.e;
    if (im.m == 0.0) e = re.e;
    return cfe_normalizar(ldexp(re.m, re.e - e), ldexp(im.m, im.e - e), e);
}

// ---------------------------------------------------------------------------
// floatexp con mantisa float (FF / CFF)
// ---------------------------------------------------------------------------
// Igual que FE / CFE pero con 24 bits de mantisa en vez de 53. Las placas GeForce hacen
// cuentas en float unas 64 veces más rápido que en double, así que en GPU es mucho más rápido;
// a cambio es menos preciso (se usa solo en los métodos "_float").

struct FF  { float m; int e; };
struct CFF { float re, im; int e; };

HD bool finito(float x) { return x - x == 0.0f; }
HD FF ff_normalizar(float m, int e) {
    if (m == 0.0f || !finito(m)) { FF r = {m, 0}; return r; }
    int k; float mm = frexpf(m, &k);
    FF r = {mm, e + k}; return r;
}
HD FF ff(double x) { FE f = fe(x); return ff_normalizar((float)f.m, f.e); }
HD FF operator*(FF a, FF b) { return ff_normalizar(a.m * b.m, a.e + b.e); }
HD FF operator/(FF a, FF b) { return ff_normalizar(a.m / b.m, a.e - b.e); }
HD FF operator+(FF a, FF b) {
    if (a.m == 0.0f) return b;
    if (b.m == 0.0f) return a;
    if (a.e >= b.e) return ff_normalizar(a.m + ldexpf(b.m, b.e - a.e), a.e);
    return ff_normalizar(b.m + ldexpf(a.m, a.e - b.e), b.e);
}
HD FF operator-(FF a, FF b) { b.m = -b.m; return a + b; }
HD double a_double(FF a) { return ldexp((double)a.m, a.e); }

HD CFF cff_normalizar(float re, float im, int e) {
    float m = fmaxf(fabsf(re), fabsf(im));
    if (m == 0.0f || !finito(m)) { CFF r = {re, im, 0}; return r; }
    int k; frexpf(m, &k);
    CFF r = {ldexpf(re, -k), ldexpf(im, -k), e + k}; return r;
}
HD CFF operator+(CFF a, CFF b) {
    if (a.re == 0.0f && a.im == 0.0f) return b;
    if (b.re == 0.0f && b.im == 0.0f) return a;
    if (a.e >= b.e) {
        float s = ldexpf(1.0f, b.e - a.e);   // 0 si la diferencia de exponentes es enorme
        return cff_normalizar(a.re + b.re * s, a.im + b.im * s, a.e);
    }
    float s = ldexpf(1.0f, a.e - b.e);
    return cff_normalizar(b.re + a.re * s, b.im + a.im * s, b.e);
}
HD CFF operator*(CFF a, CFF b) {
    return cff_normalizar(fmaf(a.re, b.re, -(a.im * b.im)), fmaf(a.re, b.im, a.im * b.re), a.e + b.e);
}
HD CFF operator-(CFF a, CFF b) { b.re = -b.re; b.im = -b.im; return a + b; }
HD CFF doble(CFF a) { a.e += 1; return a; }
HD CFF mul_sum(CFF a, CFF b, CFF c) { return a * b + c; }
HD CFF conjugado(CFF a) { a.im = -a.im; return a; }
HD CFF a_cff(CFE c) { return cff_normalizar((float)c.re, (float)c.im, c.e); }

template<> HD CFF complejo<CFF>(double re, double im) { return a_cff(cfe_normalizar(re, im, 0)); }
template<> HD FF  real<FF>(double x) { return ff(x); }
template<> HD CFF complejo_fe<CFF>(FE re, FE im) { return a_cff(complejo_fe<CFE>(re, im)); }

HD FF   norma2(CFF a) { return ff_normalizar(fmaf(a.re, a.re, a.im * a.im), 2 * a.e); }
HD FF   modulo(CFF a) { return ff_normalizar(sqrtf(fmaf(a.re, a.re, a.im * a.im)), a.e); }
HD FF   cota(CFF a)   { return ff_normalizar(1.41421356f * fmaxf(fabsf(a.re), fabsf(a.im)), a.e); }
HD bool finito(CFF a) { return finito(a.re) && finito(a.im); }
HD void direccion(CFF a, float* nx, float* ny) { direccion((double)a.re, (double)a.im, nx, ny); }

HD bool menor_pos(FF a, FF b) {
    if (b.m == 0.0f) return false;
    if (a.m == 0.0f) return true;
    if (a.e != b.e) return a.e < b.e;
    return a.m < b.m;
}
HD bool positivo(FF a) { return a.m > 0.0f && finito(a.m); }
HD bool es_cero(FF a)  { return a.m == 0.0f; }

// Un FE pasado al tipo real R de cada modo
template<class R> HD R real_de_fe(FE x);
template<> HD double real_de_fe<double>(FE x) { return a_double(x); }
template<> HD FE     real_de_fe<FE>(FE x)     { return x; }
template<> HD FF     real_de_fe<FF>(FE x)     { return ff_normalizar((float)x.m, x.e); }

// ---------------------------------------------------------------------------
// Grilla de puntos a calcular
// ---------------------------------------------------------------------------
// Rectangular (una imagen común): el punto (i, j) tiene dc = (bx + i sx, by + j sy).
// Polar (la tira del mapa exponencial): dc = radio_j (cos_i, sin_i), con el paso local
// radio_j * dtheta. Los radios y los cosenos/senos vienen en tablas calculadas una sola vez
// (en Python), así CPU y GPU usan exactamente los mismos valores y dan lo mismo bit a bit.
struct Grilla {
    FE bx, by, sx, sy, paso;
    const double* radio_m;      // polar si no es nulo (radio = radio_m * 2^radio_e)
    const int* radio_e;
    const double* cos_t;
    const double* sin_t;
    FE dtheta;
};

template<class C>
HD C dc_de_grilla(const Grilla& g, int i, int j, FE* paso)
{
    if (g.radio_m) {
        FE r = fe_normalizar(g.radio_m[j], g.radio_e[j]);
        *paso = r * g.dtheta;
        return complejo_fe<C>(r * fe(g.cos_t[i]), r * fe(g.sin_t[i]));
    }
    *paso = g.paso;
    return complejo_fe<C>(g.bx + g.sx * fe((double)i), g.by + g.sy * fe((double)j));
}

// ---------------------------------------------------------------------------
// Fórmulas
// ---------------------------------------------------------------------------
// Una fórmula es una estructura con funciones estáticas. Con la órbita de referencia Z y la
// diferencia dz de cada píxel (z = Z + dz):
//
//   dz_inicial(dc)      dz en la iteración 0
//   der_inicial()       derivada d(dz)/d(dc) en la iteración 0
//   paso(Z, dz, dc)     dz en la iteración siguiente (la perturbación)
//   derivada(z, der)    derivada en la iteración siguiente, con z el valor completo
//   paso_completo(z, c) una iteración sin perturbar (para seguir iterando después del escape)
//   c_completo(k, dc)   el "c" de ese paso_completo para el píxel (k: parámetro de la fórmula)
//   tiene_bla / bla(Z)  si se puede linealizar y con qué coeficientes: dz' ~ A dz + B dc
//
// La órbita de referencia en alta precisión (GMP) vive en perturbacion.cpp.
//
// Para agregar una fórmula: escribir su estructura acá, su paso de referencia en
// perturbacion.cpp e instanciarla en los lugares marcados con "FORMULAS".

struct Mandelbrot {
    // z' = z^2 + c, con z_0 = 0. El píxel varía c: dc.
    static const bool tiene_bla = true;
    template<class C> HD static C dz_inicial(C)  { return complejo<C>(0.0, 0.0); }
    template<class C> HD static C der_inicial()  { return complejo<C>(0.0, 0.0); }
    // dz' = (2 Z + dz) dz + dc
    template<class C> HD static C paso(C Z, C dz, C dc) { return mul_sum(doble(Z) + dz, dz, dc); }
    // der' = 2 z der + 1
    template<class C> HD static C derivada(C z, C der) { return mul_sum(doble(z), der, complejo<C>(1.0, 0.0)); }
    template<class C> HD static C paso_completo(C z, C c) { return z * z + c; }
    // k es el centro de la vista (la c de la referencia)
    template<class C> HD static C c_completo(C k, C dc) { return k + dc; }
    // El término que se descarta es dz^2: dz' ~ 2 Z dz + dc
    template<class C> HD static void bla(C Z, C* A, C* B) { *A = doble(Z); *B = complejo<C>(1.0, 0.0); }
};

// ---------------------------------------------------------------------------
// BLA
// ---------------------------------------------------------------------------
// Una entrada saltea 2^k iteraciones empezando en la posición m de la órbita:
//     dz -> A dz + B dc      válido si |dz| < r
// La tabla está organizada por niveles: el nivel k tiene cnt[k] entradas desde off[k], y la
// entrada j arranca en m = 1 + j * 2^k.

#define BLA_MAX_NIVELES 32

template<class C, class R> struct Bla { C A; C B; R r; };

// Nivel 0, un paso desde m: los coeficientes los da la fórmula. Lo descartado es despreciable
// si |dz| < eps |A|.
//
// Componer x (desde m) e y (desde m + l_x):
//     A = A_y A_x,   B = A_y B_x + B_y,   r = min(r_x, (r_y - |B_x| |dc|max) / |A_x|)
template<class F, class C, class R>
int construir_bla(const C* Z, int largo_orbita, R dc_max, double eps,
                  Bla<C, R>* tabla, int* off, int* cnt)
{
    int n0 = largo_orbita - 2;      // pasos m -> m+1 con m = 1 .. L-2
    if (!F::tiene_bla || n0 <= 0 || eps <= 0.0) return 0;

    for (int j = 0; j < n0; ++j) {
        Bla<C, R>& b = tabla[j];
        F::bla(Z[j + 1], &b.A, &b.B);
        b.r = real<R>(eps) * modulo(b.A);
    }
    off[0] = 0; cnt[0] = n0;

    int niveles = 1;
    while (niveles < BLA_MAX_NIVELES && cnt[niveles - 1] >= 2) {
        const Bla<C, R>* prev = tabla + off[niveles - 1];
        int n = cnt[niveles - 1] / 2;
        off[niveles] = off[niveles - 1] + cnt[niveles - 1];
        cnt[niveles] = n;
        Bla<C, R>* cur = tabla + off[niveles];
        for (int j = 0; j < n; ++j) {
            const Bla<C, R>& x = prev[2 * j];
            const Bla<C, R>& y = prev[2 * j + 1];
            Bla<C, R>& z = cur[j];
            z.A = y.A * x.A;
            z.B = y.A * x.B + y.B;
            R mod_ax = modulo(x.A);
            R t = real<R>(0.0);
            if (!es_cero(mod_ax)) t = (y.r - modulo(x.B) * dc_max) / mod_ax;
            if (!positivo(t)) t = real<R>(0.0);          // también descarta NaN
            z.r = menor_pos(t, x.r) ? t : x.r;
            if (!finito(z.A) || !finito(z.B)) z.r = real<R>(0.0);
        }
        ++niveles;
    }
    return niveles;
}

// Busca el salto más largo válido desde m. Devuelve el nivel, o -1 si no hay ninguno.
// r decrece al subir de nivel, así que se prueba de arriba hacia abajo.
template<class C, class R>
HD int buscar_bla(const Bla<C, R>* tabla, const int* off, const int* cnt, int niveles,
                  int m, R cota_dz, int restantes)
{
    if (m <= 0 || niveles == 0) return -1;
    int idx = m - 1;
    // Si no vale el salto de un paso no vale ninguno: es el caso más común, que salga rápido
    if (idx >= cnt[0] || !menor_pos(cota_dz, tabla[idx].r)) return -1;
    int k = niveles - 1;
    while (k > 0 && (idx & ((1 << k) - 1))) --k;   // nivel más alto en el que m está alineado
    for (; k >= 0; --k) {
        if ((1 << k) > restantes) continue;
        int j = idx >> k;
        if (j >= cnt[k]) continue;
        if (menor_pos(cota_dz, tabla[off[k] + j].r)) return k;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Iteración de un píxel
// ---------------------------------------------------------------------------

// Qué calcular además de las iteraciones (bits de `salidas`)
#define SALIDA_SUAVE    1   // iteración continua, para el coloreo suave
#define SALIDA_DERIVADA 2   // distancia al borde y normal, para la iluminación
#define DETECTAR_INTERIOR 4 // cortar antes los píxeles que caen en un ciclo (interior del conjunto)

// Detección del interior: se acumula |dz_n / dz_1|^2 = prod |2 z_k|^2. Dentro del conjunto la
// órbita cae en un ciclo atractor y ese producto tiende a 0; afuera crece. Cuando baja de este
// umbral el píxel se da por interior (el resultado es el mismo que si llegara a max_iter).
#define UMBRAL_INTERIOR2 1e-12

// En double el producto puede desbordarse en los primeros pasos (antes de caer en el ciclo):
// se lo limita, lo que solo puede demorar la detección, nunca adelantarla para un punto exterior.
HD void limitar(double& r) { if (!(r < 1e290)) r = 1e290; }
HD void limitar(FE&) {}
HD void limitar(FF&) {}

// Iteraciones extra después del escape: |z| crece como |z|^(2^k) y el valor suave sale
// tan preciso como con un radio de escape enorme, sin cambiar el conteo de iteraciones.
#define ITERACIONES_EXTRA 4

struct Pixel {
    int n;           // iteración de escape (si z_{n+1} escapa, n; si no escapa, max_iter)
    float suave;     // iteración continua (max_iter en el interior)
    float distancia; // distancia estimada al borde, en píxeles (0 en el interior)
    float nx, ny;    // normal unitaria del "relieve" (0 en el interior)
};

// Estado de un píxel a mitad de camino. La GPU lo guarda entre lanzamientos: cada lanzamiento
// avanza todos los píxeles una cantidad limitada de pasos (así ninguno dura más de lo que
// Windows tolera para un kernel, y se puede cancelar entre lanzamientos).
#define PIXEL_EN_CURSO   0
#define PIXEL_ESCAPO     1
#define PIXEL_NO_ESCAPA  2   // llegó a max_iter o se detectó que es interior

template<class C, class R> struct EstadoPixel {
    C dz, der, z;
    R z2, der_z2;
    int m, n, fin;
};

template<class F, class C, class R>
HD void iniciar_pixel(EstadoPixel<C, R>& e, const C* Z, C dc)
{
    e.dz = F::dz_inicial(dc);
    e.der = F::template der_inicial<C>();
    e.z = Z[0] + e.dz;
    e.z2 = norma2(e.z);              // |z|^2 del paso actual
    e.der_z2 = real<R>(1.0);         // |dz_n / dz_1|^2, para detectar el interior
    e.m = 0;                         // posición en la órbita de referencia
    e.n = 0;                         // iteraciones del píxel
    e.fin = PIXEL_EN_CURSO;
}

// Avanza el píxel hasta que termina o hasta hacer `max_pasos` pasos (un paso puede ser un
// salto del BLA de muchas iteraciones).
template<class F, class C, class R>
HD void avanzar_pixel(EstadoPixel<C, R>& e, const C* Z, int largo_orbita,
                      const Bla<C, R>* tabla, const int* off, const int* cnt, int niveles,
                      C dc, int max_iter, int salidas, long long max_pasos, long long* pasos)
{
    const bool con_derivada = (salidas & SALIDA_DERIVADA) != 0;
    const bool detectar_interior = (salidas & DETECTAR_INTERIOR) != 0;
    const R cuatro = real<R>(4.0);
    const R umbral_interior = real<R>(UMBRAL_INTERIOR2);
    C dz = e.dz, der = e.der, z = e.z;
    R z2 = e.z2, der_z2 = e.der_z2;
    int m = e.m, n = e.n, fin = PIXEL_EN_CURSO;
    long long p = 0;
    while (true) {
        if (n >= max_iter) { fin = PIXEL_NO_ESCAPA; break; }
        if (p >= max_pasos) break;
        int nivel = F::tiene_bla ? buscar_bla(tabla, off, cnt, niveles, m, cota(dz), max_iter - n) : -1;
        if (nivel >= 0) {
            const Bla<C, R>& b = tabla[off[nivel] + ((m - 1) >> nivel)];
            if (con_derivada) der = mul_sum(b.A, der, b.B);
            // dz/dz se transforma con A (lo mismo que dz, sin el término de dc)
            if (detectar_interior) der_z2 = der_z2 * norma2(b.A);
            dz = mul_sum(b.A, dz, b.B * dc);
            m += 1 << nivel;
            n += 1 << nivel;
        } else {
            if (con_derivada) der = F::derivada(z, der);
            // Un paso multiplica dz/dz por 2 z (z = 0 solo en el punto de partida: no cuenta)
            if (detectar_interior && !es_cero(z2)) der_z2 = der_z2 * (cuatro * z2);
            dz = F::paso(Z[m], dz, dc);
            ++m;
            ++n;
        }
        ++p;

        z = Z[m] + dz;
        z2 = norma2(z);
        if (menor_pos(cuatro, z2)) { fin = PIXEL_ESCAPO; break; }
        if (detectar_interior) {
            limitar(der_z2);
            if (menor_pos(der_z2, umbral_interior)) { fin = PIXEL_NO_ESCAPA; break; }   // ciclo atractor
        }

        // Rebasing: si la diferencia respecto del inicio de la órbita es más chica que la actual,
        // se sigue desde ahí (en Mandelbrot Z_0 = 0, así que es |z| < |dz|)
        C dz_nuevo = z - Z[0];
        if (menor_pos(norma2(dz_nuevo), norma2(dz)) || m == largo_orbita - 1) {
            dz = dz_nuevo;
            m = 0;
        }
    }
    e.dz = dz; e.der = der; e.z = z; e.z2 = z2; e.der_z2 = der_z2; e.m = m; e.n = n; e.fin = fin;
    *pasos += p;
}

// Resultado de un píxel terminado: iteración, valor suave, distancia y normal.
template<class F, class C, class R>
HD Pixel terminar_pixel(const EstadoPixel<C, R>& e, C dc, C k, int max_iter, int salidas, R paso_pixel)
{
    const bool con_derivada = (salidas & SALIDA_DERIVADA) != 0;
    Pixel px;
    px.suave = 0.0f; px.distancia = 0.0f; px.nx = 0.0f; px.ny = 0.0f;
    if (e.fin != PIXEL_ESCAPO) {
        px.n = max_iter;
        px.suave = (float)max_iter;
        return px;
    }
    px.n = e.n - 1;
    if (salidas & (SALIDA_SUAVE | SALIDA_DERIVADA)) {
        C z = e.z, der = e.der;
        C c = F::c_completo(k, dc);
        for (int i = 0; i < ITERACIONES_EXTRA; ++i) {
            if (con_derivada) der = F::derivada(z, der);
            z = F::paso_completo(z, c);
        }
        double ln_z = 0.5 * log(a_double(norma2(z)));      // ln|z|
        if (salidas & SALIDA_SUAVE) {
            // mu = n + extra - log2(ln|z|): continua entre bandas
            double mu = (double)(e.n + ITERACIONES_EXTRA) - log2(ln_z);
            px.suave = (float)(mu > 0.0 ? mu : 0.0);
        }
        if (con_derivada) {
            // Estimador de distancia: d = |z| ln|z| / |dz/dc|, pasado a píxeles
            R d = real<R>(sqrt(a_double(norma2(z))) * ln_z) / (modulo(der) * paso_pixel);
            px.distancia = (float)a_double(d);
            // Normal del relieve: dirección de z / der (= z * conj(der) / |der|^2)
            direccion(z * conjugado(der), &px.nx, &px.ny);
        }
    }
    return px;
}

// Un píxel entero, de una vez (la CPU lo usa así).
template<class F, class C, class R>
HD Pixel iterar_pixel(const C* Z, int largo_orbita,
                      const Bla<C, R>* tabla, const int* off, const int* cnt, int niveles,
                      C dc, C k, int max_iter, int salidas, R paso_pixel, long long* pasos)
{
    EstadoPixel<C, R> e;
    iniciar_pixel<F, C, R>(e, Z, dc);
    avanzar_pixel<F, C, R>(e, Z, largo_orbita, tabla, off, cnt, niveles, dc, max_iter, salidas,
                           (long long)max_iter + 1, pasos);
    return terminar_pixel<F, C, R>(e, dc, k, max_iter, salidas, paso_pixel);
}
