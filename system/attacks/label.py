import torch
import random
from torch.utils.data import DataLoader, TensorDataset

class Label:
    def __init__(self):
        pass

    def dilution(self, dataloader, target):
        dataset = dataloader.dataset  

        data_items = []
        for x, y in dataset:
            if y.ndim > 0 and y.numel() > 1:
                y_val = torch.argmax(y).item()
            else:
                y_val = y.item()
            data_items.append((x, torch.tensor(y_val, dtype=torch.long)))

        # Separa target e não-target
        target_items = [item for item in data_items if item[1].item() == target]
        other_items  = [item for item in data_items if item[1].item() != target]

        if not target_items:
            print(f"Nenhum item com classe {target} encontrado.")
            return dataloader

        # Descobre as outras classes
        other_classes = sorted(set(y.item() for _, y in other_items))
        num_classes = len(other_classes)

        # Agora vamos pegar igualmente das outras classes para compor novo target
        # per_class_take = len(target_items) // num_classes
        # gathered_for_target = []
        # updated_other_items = []

        # for cls in other_classes:
        #     class_items = [item for item in other_items if item[1].item() == cls]
        #     selected_for_target = random.sample(class_items, per_class_take)
        #     remaining_items = [item for item in class_items if item not in selected_for_target]

        #     gathered_for_target.extend((x, torch.tensor(target, dtype=torch.long)) for x, _ in selected_for_target)
        #     updated_other_items.extend(remaining_items)

        # Espalha o target nas outras classes proporcionalmente
        redistributed_target = []
        for idx, (x, _) in enumerate(target_items):
            new_class = other_classes[idx % num_classes]
            # print(f"{target} -> {new_class}")
            redistributed_target.append((x, torch.tensor(new_class, dtype=torch.long)))

        # Agora vamos pegar igualmente das outras classes para compor novo target
        total_other = sum(len([1 for _, y in other_items if y.item() == cls]) for cls in other_classes)
        gathered_for_target = []
        updated_other_items = []
        t = 0
        for cls in other_classes:
            class_items = [item for item in other_items if item[1].item() == cls]
            # print(f"tamanho da classe: {len(class_items)}")
            t += len(class_items)
            take_count = max(1, int(len(class_items) / total_other * len(target_items)))  # proporcional
            take_count = min(take_count, len(class_items))  # garante que não pega mais do que existe

            indices = list(range(len(class_items)))
            selected_indices = set(random.sample(indices, take_count))

            selected_for_target = [class_items[i] for i in selected_indices]
            remaining_items = [class_items[i] for i in indices if i not in selected_indices]

            # print(f"Classe {cls} cede {take_count} exemplos para virar target {target}.")

            gathered_for_target.extend((x, torch.tensor(target, dtype=torch.long)) for x, _ in selected_for_target)
            updated_other_items.extend(remaining_items)
        # print(f"Total de elementos: {t}")
        # Junta tudo
        final_items = redistributed_target + gathered_for_target + updated_other_items

        # Converte para tensores
        xs = torch.stack([x for x, _ in final_items])
        ys = torch.stack([y for _, y in final_items])

        # Retorna DataLoader com dataset modificado
        new_dataset = TensorDataset(xs, ys)
        return DataLoader(new_dataset, batch_size=dataloader.batch_size, shuffle=True)

    def label_flipping(self, dataloader, target, source):
        dataset = dataloader.dataset

        xs = []
        ys = []

        for x, y in dataset:
            if y.ndim > 0 and y.numel() > 1:
                y_val = torch.argmax(y).item()
            else:
                y_val = y.item()

            if y_val == target:
                y_new = torch.tensor(source, dtype=y.dtype)
            elif y_val == source:
                y_new = torch.tensor(target, dtype=y.dtype)
            else:
                y_new = y.clone()
            
            xs.append(x.unsqueeze(0))
            ys.append(y_new.unsqueeze(0))

        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)

        poisoned_dataset = TensorDataset(xs, ys)

        
        return DataLoader(poisoned_dataset,
                          batch_size=dataloader.batch_size,
                          shuffle=dataloader.shuffle if hasattr(dataloader, "shuffle") else True)