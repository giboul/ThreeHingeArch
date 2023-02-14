import numpy as np
from typing import Dict, List, Callable
from matplotlib import pyplot as plt


numpy_array = np.ndarray

colors = dict(
    arch='k',
    load='r',
    force='g',
    shear='m',
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
    yl = y[x <= L/2]
    return np.concatenate((yl, yl[::-1]))


def arch_deriv(x: numpy_array, L: float, H: float, k: float) -> numpy_array:
    d = (k*L-2*x)*4/(L*(2*k*L-L))*H
    dl = d[x <= L/2]
    return np.concatenate((dl, -dl[::-1]))


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
            vertical_load: Callable = lambda x: np.zeros_like(x),
            right_load: Callable = lambda y: np.zeros_like(y),
            left_load: Callable = lambda y: np.zeros_like(y),
            surface_load: Callable = lambda x: np.zeros_like(x),
            density: float = 0,  # N/m
            colors: dict[str: str] = colors,
            alpha: float = alpha
    ) -> None:

        # Geometry is entirely defined from these 3 parameters
        self.L = L  # Lenght
        self.H = H  # Height
        self.k0 = k  # top angle coeficient

        self.colors = colors  # Diagram & plot colors
        self.alpha = alpha  # Diagram fills transparency

        # numerical shape
        self.x = np.linspace(0, L, num=num)
        self.y = arch_shape(self.x, L, H, k)
        self.derivative = arch_deriv(self.x, L, H, k)

        # Left side
        left = self.x <= self.L/2
        self.xl = self.x[left]
        self.yl = self.y[left]

        # Rigth side
        right = self.x > self.L/2
        self.xr = self.x[right]
        self.yr = self.y[right]

        # direction vector
        self.dv = np.array(
            (np.ones_like(self.derivative), self.derivative)
        )/np.sqrt(1+self.derivative**2)
        self.dl = np.linalg.norm((np.ones(num), self.derivative), axis=0)
        # normal vector
        self.nv = np.array((-self.dv[1, :], self.dv[0, :]))
        # Attaching the essentials to the object
        self.geometry = (self.x, self.y, self.L, self.H)

        dead_weight = self.dl*density
        sf = self.dl*surface_load(self.x)
        sfx, sfy = sf*np.abs(self.nv) * self.dl

        self.v = vertical_load(self.x) + dead_weight + sfy
        self.vl = self.v[left]
        self.vr = self.v[right]
        self.hl = left_load(self.yl) + sfx[left]
        self.hr = right_load(self.yr)[::-1] + sfx[right]

        self.calculate()

    def calculate(self):
        values, arrays = self.calc_loads()
        reactions = self.calc_reactions(values)
        return self.calc_stresses(arrays, reactions)

    def calc_loads(self):

        # Vertical left load
        Vls = integrate(self.xl, self.vl)
        # Vertical left load's moment
        Mvls = integrate(self.xl, self.vl*self.xl)
        Vl = Vls[-1]
        Mvl = Mvls[-1]

        # Horizontal left load
        Hls = integrate(self.yl, self.hl)
        # Horizontal left load's moment
        Mhls = integrate(self.yl, self.hl*self.yl)
        Hl = Hls[-1]
        Mhl = Mhls[-1]

        # Vertical right load
        Vrs = integrate(self.xr, self.vr)
        # Vertical right load's moment
        Mvrs = integrate(self.xr, self.vr*(self.xr-self.L/2))
        Vr = Vrs[-1]
        Mvr = Mvrs[-1]

        # Horizontal right load
        Hrs = -integrate(self.yr, self.hr)
        # Horizontal right load's moment
        Mhrs = -integrate(self.yr, self.hr*(self.H - self.yr))
        Hr = Hrs[-1]
        Mhr = Mhrs[-1]

        return (
            # Load sums for the reaction's calculation
            (Vr, Mvr, Vl, Mvl,
             Hr, Mhr, Hl, Mhl),
            # Load arrays for moment and normal force distribution
            (Vrs, Mvrs, Vls, Mvls,
             Hrs, Mhrs, Hls, Mhls)
        )

    def calc_reactions(self, load_values):

        (Vr, Mvr, Vl, Mvl,
         Hr, Mhr, Hl, Mhl) = load_values

        A = np.array([
            # Ax  Ay       Bx         By         Cx         Cy
            [1,    0,      -1,         0,        0,         0],
            [0,    0,       1,         0,       -1,         0],
            [0,    1,       0,         1,        0,         0],
            [0,    0,       0,        -1,        0,         1],
            [0,    0,  self.H,  self.L/2,        0,         0],
            [0,    0,       0,         0,  -self.H,  self.L/2],
        ])
        b = np.array([
            -Hl,  # Ax - Bx =-Hl
            +Hr,  # Bx - Cx = Hr
            Vl,  # Ay + By = Vl
            Vr,  # -By + Cy = Vr
            Mvl+Mhl,  # Bx*H + By*L/2 = Mvl + Mhl
            Mvr+Mhr  # -Cx*H + Cy*L/2 = Mvr + Mhr
        ])

        return np.linalg.solve(A, b)

    def calc_stresses(self, load_arrays, reactions):

        (Vrs, Mvrs, Vls, Mvls,
         Hrs, Mhrs, Hls, Mhls) = load_arrays

        Ax, Ay, Bx, By, Cx, Cy = reactions

        Ml = (
            (Ax+Hls)*self.yl - (Ay-Vls)*self.xl
            - Mhls - Mvls
        )
        Mr = (
            - Mvrs - Mhrs
            - (Bx-Hrs)*(self.H-self.yr) + (By+Vrs)*(self.xr-self.L/2)
        )

        self.M = np.concatenate((Ml, Mr))

        Fx = np.concatenate((Ax+Hls, Bx-Hrs))
        Fy = np.concatenate((Vls-Ay, By+Vrs))
        F = np.array([Fx, Fy])
        self.N = (F*self.dv).sum(axis=0)
        self.V = (F*self.nv).sum(axis=0)

        return self.M, self.N, self.V

    def diagram(self,
                loads_scale=1.5,
                moments_scale=1.5,
                force_scale=4,
                show=True):

        # In order to scale the moment diagram to the maximum values
        moments_scale *= np.abs(self.M).max()
        force_scale *= np.abs(self.N).max()
        loads_scale *= max(
            np.abs(load).max() for load in (self.v, self.hr, self.hl)
        )

        # Check if it is null to avoid divisions by 0
        loads_scale = 1 if loads_scale == 0 else loads_scale
        moments_scale = 1 if moments_scale == 0 else moments_scale
        force_scale = 1 if force_scale == 0 else force_scale

        plt.axis('off')
        # Arch
        plt.plot(self.x, self.y, '-k', lw=3)
        # Hinges
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
        Vx, Vy = self.V*self.nv/force_scale
        normal, = plt.fill(
            np.concatenate((self.x+Nx, self.x[::-1])),
            np.concatenate((self.y+Ny, self.y[::-1])),
            edgecolor='none',
            color=colors['force'],
            label="Normal force",
            alpha=alpha
        )
        shear, = plt.fill(
            np.concatenate((self.x+Vx, self.x[::-1])),
            np.concatenate((self.y+Vy, self.y[::-1])),
            edgecolor='none',
            color=colors['shear'],
            label="Shear force",
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
                    1.1*self.L + self.hr[::-1]/loads_scale,
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

        # Grouping labels into the axis' legend
        if load is None:
            lines = (moment, normal, shear)
            print("Warning: no loads found")
        else:
            lines = (moment, normal, shear, load)
        labs = [line.get_label() for line in lines]
        plt.legend(lines, labs)

        if show:
            plt.show()

    def plot(self, show_loads=True, show=True):

        fig, ax1 = plt.subplots(1, sharex=True, figsize=(8, 6))
        ax2 = ax1.twinx()
        ax1.axline((0, 0), slope=0, color='k', linestyle='--', lw=1)
        ax2.axline((0, 0), slope=0, color='k', linestyle='--', lw=1)
        moment, = ax1.plot(self.x, self.M, '--', label="Moments",
                           color=colors['moment'], lw=2)
        normal, = ax2.plot(self.x, self.N, '-.', label="Normal force",
                           color=colors['force'], lw=1.5)
        shear, = ax2.plot(self.x, self.V, '-.', label="Shear force",
                          color=colors['shear'], lw=1.5)

        if show_loads:
            lns3, = ax2.plot(self.x, self.v, '-.',
                             color=colors['load'],
                             label='Vertical load', lw=1)
            lns4, = ax2.plot(self.x, np.concatenate((self.hl, self.hr)), ':',
                             color=colors['load'],
                             label='Horizontal load', lw=1)
            lns = (moment, normal, shear, lns3, lns4)
        else:
            lns = (moment, normal, shear)

        # Merging labels into one legend
        labs = [ln.get_label() for ln in lns]
        ax2.legend(lns, labs, loc=(0.01, 1.0), ncols=4)
        ax1.set_xlabel("Position [m]")
        ax1.set_ylabel("Moment [N$\\cdot$m]", color=colors["moment"])
        ax2.set_ylabel("Force [N]")

        # Detail colors
        ax1.tick_params(colors=colors["moment"], axis="y")

        if show:
            plt.show()


if __name__ == "__main__":

    def horizontal_load_right(x: numpy_array):
        return np.zeros_like(x)

    def vertical_load(x: numpy_array):
        return (x-x.min())/(x.max()-x.min())

    def horizontal_load_left(x: numpy_array):
        return np.exp(vertical_load(x))*0

    def surface_load(x: numpy_array):
        return np.ones_like(x)

    arch = ThreeHingedArch(
        L=4,  # length
        H=3,  # height
        k=1.0,
        surface_load=surface_load,
        # vertical_load=vertical_load,
        # right_load=horizontal_load_right,
        # left_load=horizontal_load_left
    )
    arch.diagram()
    plt.style.use('bmh')
    arch.plot(show_loads=False)
