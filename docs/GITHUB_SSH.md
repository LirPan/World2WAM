# Push to GitHub via SSH

Repository: https://github.com/LirPan/World2WAM

## 1. Add SSH public key (one-time)

On the server:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output, then on GitHub:

1. Open [SSH and GPG keys](https://github.com/settings/keys)
2. **New SSH key** → paste the public key → Save

Test:

```bash
ssh -T git@github.com
# Hi LirPan! You've successfully authenticated...
```

## 2. Push

```bash
cd /DATA/disk1/yjh_space/idea2_workspace/minimal_world2wam
git push -u origin main
```

Remote is already set to `git@github.com:LirPan/World2WAM.git`.
