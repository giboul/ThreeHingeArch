import numpy as np
from typing import Dict, List, Callable
from matplotlib import pyplot as plt


numpy_array = np.ndarray

colors = dict(
    arch='k',
    load='r',
    force='g',
    moment='b'
)
alpha = 0.3


def integrate(
        x: numpy_array,
        y: numpy_array,
        c: float = 0,
        expand: bool = True,
        cumul: bool = True) -> numpy_array:

    arr = np.cumsum((x[1:]-x[:-1])*(y[1:]+y[:-1]))/2 + c
    if expand:
        arr = np.concatenate(((c,), arr))
    if not cumul:
        return arr[-1]
    return arr


def arch_shape(x: numpy_array, L: float, H: float, k: float) -> numpy_array:
    y = x*(k*L-x)*4/(L*(2*k*L-L))*H
    yr = y[x <= L/2]
    return np.concatenate((yr, yr[::-1]))


def arch_deriv(x: numpy_array, L: float, H: float, k: float) -> numpy_array:
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
            num: int = 1000,
            vertical_load: Callable = lambda x: np.exp(x),
            right_load: Callable = lambda x: (x-x.min())/(x.max()-x.min()),
            left_load: Callable = lambda x: np.ones_like(x),
    ) -> None:

        self.L = L
        self.H = H
        self.k0 = k

        self.x = np.linspace(0, L, num=num)
        self.y = arch_shape(self.x, L, H, k)
        self.derivative = arch_deriv(self.x, L, H, k)
        self.nv = np.array(
            (-self.derivative, np.ones_like(self.derivative))
        )/np.sqrt(1+self.derivative**2)
        self.geometry = (self.x, self.y, self.L, self.H)

        self.v = vertical_load(self.x)
        self.hl = left_load(self.y[self.x <= L/2])
        self.hr = right_load(self.y[self.x > L/2])
        self.hr = self.hr[::-1]
        self.h = np.concatenate((self.hl, self.hr))

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

        Nl = np.linalg.norm(np.array((
            Ax + Hls, Ay - Vls
        )), axis=0)
        Nr = np.linalg.norm(np.array((
            Bx - Hrs, By + Vrs
        )), axis=0)
        self.N = np.concatenate((Nl, Nr))

    def diagram(self,
                loads_scale=1.5,
                moments_scale=1.5,
                force_scale=4):

        loads_scale *= max(
            np.abs(load).max() for load in (self.v, self.hr, self.hl)
        )
        moments_scale *= np.abs(self.M).max()
        force_scale *= np.abs(self.N).max()

        loads_scale = 1 if loads_scale == 0 else loads_scale
        moments_scale = 1 if moments_scale == 0 else moments_scale
        force_scale = 1 if force_scale == 0 else force_scale

        plt.plot(self.x, self.y, '-k', lw=3)
        plt.scatter((0, self.L/2, self.L), (0, self.H, 0), s=50,
                    facecolor='w', linewidths=2, edgecolors='k', zorder=3)

        Mx, My = self.M*self.nv/moments_scale
        moment, = plt.fill(
            np.concatenate((self.x+Mx, self.x[::-1])),
            np.concatenate((self.y+My, self.y[::-1])),
            edgecolor='none',
            color=colors['moment'],
            label="Moment",
            alpha=alpha
        )

        Nx, Ny = self.N*self.nv/force_scale
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
                    1.2*self.H + self.v/loads_scale,
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
                    -0.1*self.L - self.hl/loads_scale,
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
                    1.1*self.L + self.hr/loads_scale,
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
        plt.legend(lines, labs, loc='lower center')
        plt.show()

    def plot(self):

        fig, ax1 = plt.subplots(1, sharex=True, figsize=(8, 6))
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
        ax2.legend(lns, labs, loc=(0.01, 1.0), ncols=4)
        ax1.set_xlabel("Position [m]")
        ax1.set_ylabel("Moment [N$\\cdot$m]", color=colors["moment"])
        ax2.set_ylabel("Force [N]", color=colors["force"])
        ax2.spines['left'].set_color(colors["moment"])
        ax2.spines["right"].set_color(colors["force"])
        ax1.tick_params(colors=colors["moment"], axis="y")
        ax2.tick_params(colors=colors["force"], axis="y")
        plt.show()


if __name__ == "__main__":

    def horizontal_load_right(x: numpy_array):
        return np.zeros_like(x)

    def horizontal_load_left(x: numpy_array):
        return (x-x.min())/(x.max()-x.min())

    def vertical_load(x: numpy_array):
        return np.exp(horizontal_load_left(x))

    arch = ThreeHingedArch(
        L=4,  # length
        H=3,  # height
        vertical_load=vertical_load,
        right_load=horizontal_load_right,
        left_load=horizontal_load_left
    )
    arch.diagram()
    plt.style.use('bmh')
    arch.plot()
