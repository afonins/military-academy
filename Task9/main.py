


from simulation import BrigadeSimulation
from terrain import TerrainMap


def main():
    print("="*70)
    print("СИМУЛЯЦИЯ БРИГАДНОЙ ОПЕРАЦИИ")
    print("="*70)
    print("\nСостав: 3 батальона (по 3 роты) + артиллерия + авиация + разведка")
    print("Противник: 5 опорных пунктов, минные поля, артиллерия")
    print("Местность: город, лес, равнина, река, высоты")
    print("Время: 24 часа (ускоренный режим)")
    print("="*70)
    
    # Показать карту
    terrain = TerrainMap()
    terrain.print_map() 
    
    input("\nНажмите Enter для начала симуляции...")
    
    # Запуск
    sim = BrigadeSimulation()
    sim.run()
    
    print("\n" + "="*70)
    print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
    print("="*70)


if __name__ == "__main__":
    main()