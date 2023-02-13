import numpy as np
from typing import Dict, List
from matplotlib import pyplot as plt

colors = dict(
    arch='k',
    load='r',
    force='g',
    moment='b'
)
alpha = 0.3


def integrate(
            x: np.ndarray,
            y: np.ndarray,
            c: float = 0,
            expand: bool = True,
            cumul: bool = True) -> np.ndarray:

    arr = np.cumsum((x[1:]-x[:-1])*(y[1:]+y[:-1]))/2 + c
    if expand:
        arr = np.concatenate(((c,), arr))
    if not cumul:
        return arr[-1]
    return arr


def arch_shape(x: np.ndarray, L: float, H: float, k: float) -> np.ndarray:
    return x*(k*L-x)*4/(L*(2*k*L-L))*H


def arch_deriv(x: np.ndarray, L: float, H: float, k: float) -> np.ndarray:
    return (k*L-2*x)*4/(L*(2*k*L-L))*H


class ThreeHingedArch:

    #          v +----+----+----+----+----+
    #            |    |    |    |    |    |
    #            ▼    ▼    ▼    ▼    ▼    ▼
    #                     By ▲
    #  hl                    |                        hr
    # +->               Bx <- O -> Bx                <-+
    # |                  #     |   #                  |
    # +->             #        ▼      #             <-+
    # |             #          By       #             |
    # +->          #                     #          <-+
    # |           #                       #           |
    # +->   Ax -> O                       O <- Cx   <-+
    #             ▲                       ▲
    #             |                       |
    #             Ay                      Cy

    def __init__(
            self,
            L: float = 2,
            H: float = 1,
            k: float = 1,
            num: int = None,
            loads: tuple[np.ndarray] = None,
            loads_scale=5,
            moments_scale=20,
            force_scale=20
            ) -> None:

        if num is None and loads is None:
            num = 1000
        elif num is None:
            num = len(loads[0])
        if loads is None:
            loads = [
                np.exp(np.linspace(0, 1, num)),
                np.linspace(0, 1, num//2),
                np.ones(num//2)
            ]

        self.L = L
        self.H = H
        self.k0 = k

        self.moments_scale = moments_scale
        self.force_scale = force_scale

        self.x = np.linspace(0, L, num=num)
        self.y = arch_shape(self.x, L, H, k)
        self.derivative = arch_deriv(self.x, L, H, k)
        self.nv = np.array(
                        (-self.derivative, np.ones_like(self.derivative))
                    )/np.sqrt(1+self.derivative**2)
        self.geometry = (self.x, self.y, self.L, self.H)

        self.v, self.hl, self.hr = loads
        self.hr = self.hr[::-1]
        self.h = np.concatenate((self.hl, self.hr))
        self.loads_scale = max(load.max() for load in loads)*loads_scale/L

        self.calc_constants()
        self.calc_reactions()
        self.calc_stresses()

    def calc_constants(self):

        x, y, L, H = self.geometry
        v = self.v

        # Left arch
        left = x <= L/2
        self.xl = x[left]
        self.yl = y[left]
        self.vl = v[left]
        Vls = integrate(self.xl, self.vl)
        Vls_cls = integrate(self.xl, self.vl*self.xl)
        Vl = Vls[-1]
        Vl_cl = Vls_cls[-1]
        Hls = integrate(self.yl, self.hl)
        Hls_cls = integrate(self.yl, self.hl*self.yl)
        Hl = Hls[-1]
        Hl_cl = Hls_cls[-1]

        # Rigth arch
        right = x > L/2
        self.xr = x[right]
        self.yr = y[right]
        self.vr = v[right]
        Vrs = integrate(self.xr - L/2, self.vr)
        Vrs_crs = integrate(self.xr - L/2, self.vr*(self.xr-L/2))
        Vr = Vrs[-1]
        Vr_cr = Vrs_crs[-1]
        Hrs = integrate(self.H - self.yr, self.hr)
        Hrs_crs = integrate(self.H - self.yr, self.hr*(self.H - self.yr))
        Hr = Hrs[-1]
        Hr_cr = Hrs_crs[-1]

        V = integrate(x, v, cumul=False)
        H = Hl - Hr

        self.constants = (
            (Vr, Vrs, Vr_cr, Vrs_crs),
            (Vl, Vls, Vl_cl, Vls_cls),
            (Hr, Hrs, Hr_cr, Hrs_crs),
            (Hl, Hls, Hl_cl, Hls_cls),
            (V, H)
        )
        return self.calc_constants

    def calc_reactions(self):

        (
            (Vr, Vrs, Vr_cr, Vrs_crs),
            (Vl, Vls, Vl_cl, Vls_cls),
            (Hr, Hrs, Hr_cr, Hrs_crs),
            (Hl, Hls, Hl_cl, Hls_cls),
            (V, H)
        ) = self.constants

        A = np.array([
            # Ax  Ay       Bx         By         Cx         Cy
            [1,    0,      -1,         0,        0,         0],
            [0,    0,       1,         0, -      1,         0],
            [0,    1,       0,         1,        0,         0],
            [0,    0,       0,        -1,        0,         1],
            [0,    0,  self.H,  self.L/2,        0,         0],
            [0,    0,       0,         0,  -self.H,  self.L/2],
        ])
        b = np.array([
            -Hl,
            +Hr,
            Vl,
            Vr,
            Vl_cl+Hl_cl,
            Vr_cr+Hr_cr
        ])
        self.reactions = np.linalg.solve(A, b)

        print(f"A       = {A}")
        print(f"b       = {b}")
        print(f"R       = {self.reactions}")
        print(f"Forces  = {[V, Vl, Vr, H, Hl, Hr]}")
        print(f"Moments = {[Vl_cl, Vr_cr, Hl_cl, Hr_cr]}")

        return self.reactions

    def calc_stresses(self):

        (
            (Vr, Vrs, Vr_cr, Vrs_crs),
            (Vl, Vls, Vl_cl, Vls_cls),
            (Hr, Hrs, Hr_cr, Hrs_crs),
            (Hl, Hls, Hl_cl, Hls_cls),
            (V, H)
        ) = self.constants
        Ax, Ay, Bx, By, Cx, Cy = self.reactions

        Ml = (Ax+Hls)*self.yl - (Ay-Vls)*self.xl - Hls_cls - Vls_cls
        # Mr = (
        #     Vrs_crs + Hls_cls
        #     - By*(self.xr-self.L/2) - Bx*(self.H-self.yr)
        # )
        Mr = (
            - Vrs_crs - Hrs_crs
            - (Bx-Hrs)*(self.H-self.yr) + (By+Vrs)*(self.xr-self.L/2)
        )

        self.M = np.concatenate((Ml, Mr))
        self.moments_scale = max(self.M)*self.moments_scale/self.L
        if np.isclose(self.moments_scale, 0):
            self.moments_scale = 1

        Nl = np.linalg.norm(np.array((
            Ax + Hls, Ay - Vls
        )), axis=0)
        Nr = np.linalg.norm(np.array((
            Bx - Hrs, By + Vrs
        )), axis=0)
        self.N = np.concatenate((Nl, Nr))
        self.force_scale = max(self.N)*self.force_scale/self.L

    def diagram(self):

        plt.plot(self.x, self.y, '-k', lw=3)
        plt.scatter((0, self.L/2, self.L), (0, self.H, 0), s=50,
                    facecolor='w', linewidths=2, edgecolors='k', zorder=3)
        if self.moments_scale == 0:
            self.moments_scale = 1
        Mx, My = self.M*self.nv/self.moments_scale
        moment, = plt.fill(
            np.concatenate((self.x+Mx, self.x[::-1])),
            np.concatenate((self.y+My, self.y[::-1])),
            edgecolor='none',
            color=colors['moment'],
            label="Moment",
            alpha=alpha
        )
        Nx, Ny = self.N*self.nv/self.force_scale
        normal, = plt.fill(
            np.concatenate((self.x+Nx, self.x[::-1])),
            np.concatenate((self.y+Ny, self.y[::-1])),
            edgecolor='none',
            color=colors['force'],
            label="Normal force",
            alpha=alpha
        )
        load = None
        if not (self.v == 0).all():
            load, = plt.fill(
                np.concatenate((self.x, self.x[::-1])),
                np.concatenate((
                        1.2*self.H + self.v/self.loads_scale,
                        1.2*self.H + np.zeros_like(self.v)
                    )),
                edgecolor='none',
                color=colors['load'],
                label="Loads",
                alpha=alpha
                )
        if not (self.hl == 0).all():
            load, = plt.fill(
                np.concatenate((
                        -0.1*self.L - self.hl/self.loads_scale,
                        -0.1*self.L + np.zeros_like(self.hl)
                    )),
                np.concatenate((
                        self.yl, self.yl[::-1]
                    )),
                edgecolor='none',
                color=colors['load'],
                label="Loads",
                alpha=alpha
                )
        if not (self.hr == 0).all():
            load, = plt.fill(
                np.concatenate((
                        1.1*self.L + self.hr/self.loads_scale,
                        1.1*self.L + np.zeros_like(self.hl)
                    )),
                np.concatenate((
                        self.yl, self.yl[::-1]
                    )),
                edgecolor='none',
                color=colors['load'],
                label="Loads",
                alpha=alpha
                )
        lines = [moment, normal] + [load]
        labs = [line.get_label() for line in lines]
        plt.legend(lines, labs)
        plt.show()

    def plot(self):

        fig, ax1 = plt.subplots(1, sharex=True)
        ax2 = ax1.twinx()
        ax1.axline((0, 0), slope=0, color='k', linestyle='--', lw=1)
        ax2.axline((0, 0), slope=0, color='k', linestyle='--', lw=1)
        lns1, = ax1.plot(self.x, self.M, '--', label="Moments",
                         color=colors['moment'], lw=2)
        lns2, = ax2.plot(self.x, self.N, '--', label="Normal force",
                         color=colors['force'], lw=2)
        lns3, = ax2.plot(self.x, self.v, '-.',
                         color=colors['load'], label='Vertical load', lw=1)
        lns4, = ax2.plot(self.x, self.h, ':',
                         color=colors['load'], label='Horizontal load', lw=1)

        lns = (lns1, lns2, lns3, lns4)
        labs = [ln.get_label() for ln in lns]
        ax2.legend(lns, labs, loc='center left')
        ax1.set_xlabel("Position [m]")
        ax1.set_ylabel("Moment [N$\\cdot$m]", color=colors["moment"])
        ax2.set_ylabel("Force [N]", color=colors["force"])
        ax2.spines['left'].set_color(colors["moment"])
        ax2.spines["right"].set_color(colors["force"])
        ax1.tick_params(colors=colors["moment"], axis="y")
        ax2.tick_params(colors=colors["force"], axis="y")
        plt.show()


if __name__ == "__main__":

    arch = ThreeHingedArch()
    arch.diagram()
    plt.style.use('bmh')
    arch.plot()
