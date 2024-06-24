from models_continuous import *
from scipy.integrate import solve_ivp

d = 64
n = 32
random_init = False

n_runs = 2**10

delta = 1e-3

hidden_dim = d

n_grid_t = 100
t_min, t_max = 0., 30.
list_t = torch.linspace(t_min, t_max, n_grid_t).to(device)
n_grid_beta = 100
beta_min, beta_max = 0., 9.
list_betas = np.linspace(beta_min, beta_max, n_grid_beta)

array_are_clustered = np.zeros((n_grid_beta, n_grid_t)) # betas * t
list_index_t_transition = []

for i, beta in enumerate(tqdm(list_betas, desc = 'Beta', total = len(list_betas))):
    model = FullModel(d, hidden_dim, beta, random_init = random_init, n = n).to(device)

    x = torch.randn(n_runs, n, d).to(device)
    x /= torch.norm(x, dim = 2, keepdim = True)

    res = model(x, list_t) # n_grid, n_runs, n, d

    temp = torch.einsum('abcd,abed->abce', res, res) # n_grid, n_runs, n, n
    temp = temp.reshape(n_grid_t, n_runs, n**2) # n_grid, n_runs, n*n
    temp = (temp >= 1 - delta).float() # n_grid, n_runs, n*n
    temp = (temp.sum(dim = 2) - n) / (n * (n - 1)) # n_grid, n_runs
    dot_product_res1_res2 = temp

    dot_product_res1_res2 = dot_product_res1_res2.mean(dim = 1) # n_grid

    array_are_clustered[i, :] += dot_product_res1_res2.cpu().numpy()

    gamma_beta = solve_ivp(dynamic_gamma_beta, [0, t_max], np.array([0.]), args = (beta, n), t_eval = list_t.cpu().numpy(), rtol=1e-6, atol=1e-9).y[0]
    index_t_transition = np.argmax(gamma_beta >= 1 - delta)
    if gamma_beta[index_t_transition] < 1 - delta:
        index_t_transition = -1.
    list_index_t_transition.append(index_t_transition)

plt.imshow(array_are_clustered, cmap='RdBu_r', interpolation = 'nearest', origin='lower')

max_len = np.argmax(np.array(list_index_t_transition) == -1)
plt.plot(list_index_t_transition[:max_len], np.arange(n_grid_beta)[:max_len], 'k-')

plt.xticks(np.linspace(0, n_grid_t, 5), np.round(np.linspace(t_min, t_max, 5), 2))
plt.yticks(np.linspace(0, n_grid_beta, 5), np.round(np.linspace(beta_min, beta_max, 5), 2))
plt.colorbar()
plt.xlabel(r"$t$")
plt.ylabel(r"$\beta$")
plt.title(f'Transition phase (d = {d})')
plt.savefig(f'Phase_portrait_d_{str(d)}.png')
