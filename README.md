
## Introduction


Define the longitudinal domain as $x\in[-(L_{\mathrm{cav}}/2+L_{\mathrm{ref}}),\,L_{\mathrm{cav}}/2+L_{\mathrm{ref}}]$. Inside the cavity region $|x|\le L_{\mathrm{cav}}/2$, the nanoparticle thickness profile is symmetric and parabolic:


$l(x)=l_{\min}+(l_{\max}-l_{\min})\left(1-\left(\frac{2x}{L_{\mathrm{cav}}}\right)^2\right).$


Inside the reflector regions, the modulation alternates with period $\Lambda$ and remains bounded by the same thickness interval. A non-uniform grid is required because the cavity center and the cavity-reflector interfaces are the regions where the response is most sensitive to geometric perturbations.

Let $x_1,\dots,x_N$ be the non-uniform grid. Define an effective scalar cavity operator acting on a longitudinal field envelope $u(x,\lambda)$ by a Helmholtz-type form

$\mathcal{A}(\lambda)u=-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right)+V(x,\lambda)u,$

with positive coefficient $a(x)$ and wavelength-dependent potential $V(x,\lambda)$ determined by the thickness profile, the target resonance wavelength, and the reference waveguide background. The cavity response is represented through the action of the inverse operator on localized source profiles centered at dipole positions in the cavity. The total emitted power and guided power are obtained from nonnegative quadratic forms of the cavity response. A reference response is computed from the same geometry with resonant modulation removed.

The required invariants are: symmetry of the longitudinal profile about $x=0$, nonnegativity of the returned power densities, and identical wavelength sampling for resonant and reference responses. The key numerical pathology is that the operator becomes highly ill-conditioned near the narrow resonance and the non-uniform grid must still preserve symmetry and positivity. Dense matrix construction is disallowed in the core operator build because the grid may be large.
