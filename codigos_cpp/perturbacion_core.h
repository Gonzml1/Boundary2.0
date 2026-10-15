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
  using std::frexp; using std::ldexp; using std::fabs; using std::fmax; using std::sqrt;
  using std::log; using std::log2;
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
HD CD operator*(CD a, CD b) { CD r = {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; return r; }
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
HD CFE conjugado(CFE a) { a.im = -a.im; return a; }

// --- Operaciones genéricas: cada tipo complejo C tiene un tipo real R asociado ---
// norma2 = |z|^2, modulo = |z|, cota = sqrt(2) * max(|re|, |im|) >= |z| (barata y sin underflow)
HD double norma2(CD a) { return a.re * a.re + a.im * a.im; }
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
    template<class C> HD static C paso(C Z, C dz, C dc) { return (doble(Z) + dz) * dz + dc; }
    // der' = 2 z der + 1
    template<class C> HD static C derivada(C z, C der) { return doble(z) * der + complejo<C>(1.0, 0.0); }
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

// Iteraciones extra después del escape: |z| crece como |z|^(2^k) y el valor suave sale
// tan preciso como con un radio de escape enorme, sin cambiar el conteo de iteraciones.
#define ITERACIONES_EXTRA 4

struct Pixel {
    int n;           // iteración de escape (si z_{n+1} escapa, n; si no escapa, max_iter)
    float suave;     // iteración continua (max_iter en el interior)
    float distancia; // distancia estimada al borde, en píxeles (0 en el interior)
    float nx, ny;    // normal unitaria del "relieve" (0 en el interior)
};

template<class F, class C, class R>
HD Pixel iterar_pixel(const C* Z, int largo_orbita,
                      const Bla<C, R>* tabla, const int* off, const int* cnt, int niveles,
                      C dc, C k, int max_iter, int salidas, R paso_pixel, long long* pasos)
{
    const bool con_derivada = (salidas & SALIDA_DERIVADA) != 0;
    const R cuatro = real<R>(4.0);
    C dz = F::dz_inicial(dc);
    C der = F::template der_inicial<C>();
    C z = Z[0] + dz;
    int m = 0;   // posición en la órbita de referencia
    int n = 0;   // iteraciones del píxel
    bool escapo = false;
    long long p = 0;
    while (n < max_iter) {
        int nivel = F::tiene_bla ? buscar_bla(tabla, off, cnt, niveles, m, cota(dz), max_iter - n) : -1;
        if (nivel >= 0) {
            const Bla<C, R>& b = tabla[off[nivel] + ((m - 1) >> nivel)];
            if (con_derivada) der = b.A * der + b.B;
            dz = b.A * dz + b.B * dc;
            m += 1 << nivel;
            n += 1 << nivel;
        } else {
            if (con_derivada) der = F::derivada(z, der);
            dz = F::paso(Z[m], dz, dc);
            ++m;
            ++n;
        }
        ++p;

        z = Z[m] + dz;
        R z2 = norma2(z);
        if (menor_pos(cuatro, z2)) { escapo = true; break; }

        // Rebasing: si la diferencia respecto del inicio de la órbita es más chica que la actual,
        // se sigue desde ahí (en Mandelbrot Z_0 = 0, así que es |z| < |dz|)
        C dz_nuevo = z - Z[0];
        if (menor_pos(norma2(dz_nuevo), norma2(dz)) || m == largo_orbita - 1) {
            dz = dz_nuevo;
            m = 0;
        }
    }
    *pasos += p;

    Pixel px;
    px.suave = 0.0f; px.distancia = 0.0f; px.nx = 0.0f; px.ny = 0.0f;
    if (!escapo) {
        px.n = max_iter;
        px.suave = (float)max_iter;
        return px;
    }
    px.n = n - 1;
    if (salidas & (SALIDA_SUAVE | SALIDA_DERIVADA)) {
        C c = F::c_completo(k, dc);
        for (int i = 0; i < ITERACIONES_EXTRA; ++i) {
            if (con_derivada) der = F::derivada(z, der);
            z = F::paso_completo(z, c);
        }
        double ln_z = 0.5 * log(a_double(norma2(z)));      // ln|z|
        if (salidas & SALIDA_SUAVE) {
            // mu = n + extra - log2(ln|z|): continua entre bandas
            double mu = (double)(n + ITERACIONES_EXTRA) - log2(ln_z);
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
