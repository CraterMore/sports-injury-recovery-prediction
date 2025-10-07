def main():
    # get input from user
    print("\nWelcome to the Sports Medicine Team's Injury Recovery Predictor.\nPlease follow the prompts to get your "
          "predicted recovery time, treatment method, and cost of recovery.")

    age = input("What is the athlete's age? ") # might have to change data type
    gender = input("What is the athlete's gender? ")
    minutes_played = input("How many minutes has the athlete played in the last 30 days? ")
    body_part = input("What body part did the injury occur on? ")
    injury_type = input("What is the athlete's injury (e.g. sprain, fracture, strain)? ")
    cause = input("What was the cause of the injury? ")
    reccurent = input("Is this a recurrent injury? ")
    sport = input("What sport was the athlete playing when the injury occurred? ")
    competition = input("What competition level does the athlete play at (Professional or Amateur)? ")
    region = input("What region did the injury occur in (Northern, Central, Southern, or Western Europe)? ")
    event_type = input("What event type did the injury occur during (warm-up, training, or competition)? ")
    surface = input("What surface did the injury occur on? ")

    # learn to use flask for the input so i can do radioboxes and such



    # put input through regression model

    # print output (prediction)

    # print error


if __name__ == "__main__":
    main()